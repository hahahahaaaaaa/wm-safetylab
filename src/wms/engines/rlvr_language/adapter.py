"""Official RLVR-World Bytesized32 predictor. No environment action is executed."""
from __future__ import annotations
import copy
import importlib.util
import json
import re
import time
from pathlib import Path

from wms.core.world_model import validate_request, write_prediction_bundle

MODEL_ID = 'rlvr-language-bytesized32-task-specific'
HF_ID = 'thuml/bytesized32-world-model-rlvr-task-specific-reward'
HF_REVISION = '153dd625edde0fef09bad468235c14019d91ce3f'
SOURCE_COMMIT = 'e1b8b6f40ca0696919ce9b8dbe965c4153bcf5f1'
INPUT_KEYS = {'messages', 'current_state', 'action', 'dataset_reference'}

def validate_language_request(request, manifest):
    validate_request(request, manifest)
    if request['horizon'] != {'steps': 1, 'unit': 'transition'}:
        raise ValueError('RLVR Bytesized32 adapter supports one symbolic transition only')
    inputs = request['inputs']
    if not {'messages', 'current_state', 'action'} <= set(inputs):
        raise ValueError('Language prediction needs messages, current_state and action')
    messages = inputs['messages']
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages must contain the official context-only chat prompt')
    for msg in messages:
        if not isinstance(msg, dict) or set(msg) != {'role', 'content'}:
            raise ValueError('Invalid prompt message')
        if msg['role'] not in {'user', 'system'} or not isinstance(msg['content'], str) or not msg['content']:
            raise ValueError('Only nonempty user/system prompt messages are accepted')
    if not isinstance(inputs['current_state'], dict) or not isinstance(inputs['current_state'].get('game_state'), list):
        raise ValueError('current_state requires game_state list')
    action = inputs['action']
    if not isinstance(action, str) or not action or not any(action in msg['content'] for msg in messages):
        raise ValueError('Declared action must occur in the native prompt')
    parameters = request.get('parameters', {})
    if set(parameters) - {'max_new_tokens', 'do_sample'}:
        raise ValueError('Unsupported generation parameter')
    if parameters.get('do_sample', False) is not False:
        raise ValueError('Initial adapter uses greedy decoding; sampling requires a separate protocol')
    if type(parameters.get('max_new_tokens', 4096)) is not int or not 1 <= parameters.get('max_new_tokens', 4096) <= 4096:
        raise ValueError('max_new_tokens must be in [1, 4096]')
    return request


def extract_answer(raw_text):
    """Match official final-answer prefix precedence; never silently strip reasoning."""
    for prefix in ('The final answer is:', 'the final answer is:', 'The final state is:',
                   'the final state is:', 'The final game state is:', 'the final game state is:'):
        if prefix in raw_text:
            return raw_text.split(prefix)[-1].strip()
    matches = re.findall(r'```json\s*([\s\S]*?)\s*```', raw_text, re.DOTALL)
    return matches[-1] if matches and len(matches[-1]) >= 10 else raw_text


def parse_partial(raw_text, *, allow_repair=False):
    answer = extract_answer(raw_text)
    details = {'strict_json_valid': False, 'repair_applied': False, 'parse_status': 'error'}
    try:
        partial = json.loads(answer)
        details['strict_json_valid'] = True
    except json.JSONDecodeError:
        if not allow_repair:
            raise ValueError('Model output is not strict JSON')
        from json_repair import repair_json
        import dirtyjson
        normalized = answer.replace('True', 'true').replace('False', 'false').replace('None', 'null')
        repaired = repair_json(normalized)
        partial = dirtyjson.loads(repaired)
        details['repair_applied'] = True
    if not isinstance(partial, dict) or not {'modified', 'removed'} <= set(partial):
        raise ValueError('Native state difference must contain modified and removed')
    if not isinstance(partial['modified'], list) or not isinstance(partial['removed'], list):
        raise ValueError('modified and removed must be lists')
    if any(not isinstance(obj, dict) or 'uuid' not in obj for obj in partial['modified']):
        raise ValueError('Each modified object must have uuid')
    if any(type(uid) is not int for uid in partial['removed']):
        raise ValueError('removed must contain integer UUIDs')
    if 'score' not in partial or not isinstance(partial['score'], dict):
        raise ValueError('Native prediction must include a score object')
    details['parse_status'] = 'ok'
    return partial, details


def load_official_scorer(source_dir):
    path = Path(source_dir) / 'lang_wm/verl/verl/utils/reward_score/text_game.py'
    spec = importlib.util.spec_from_file_location('wms_rlvr_official_text_game', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Missing official RLVR scoring source')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


class RLVRLanguageAdapter:
    def __init__(self, config_dict):
        self.config = dict(config_dict)
        self._tokenizer = None
        self._model = None
        self._resource_provenance = None

    def describe(self):
        return {'schema_version': 'wms.model/1.0', 'model_id': MODEL_ID,
                'name': 'RLVR-World Bytesized32 task-specific-reward language world model',
                'families': ['observation_generative_interactive'],
                'representations': ['symbolic_state'], 'capabilities': ['predict'],
                'action_conditioning': 'text_action',
                'limitations': ['One-step text-game prediction only; no robot or browser action execution.',
                    'Native output is a state difference; reconstructed state uses the unchanged official recovery function.',
                    'Greedy Transformers smoke inference is not a full verl/vLLM paper evaluation.',
                    'Prediction validity or task correctness is not a safety verdict.'],
                'provenance': {'repository': 'https://github.com/thuml/RLVR-World',
                    'source_commit': SOURCE_COMMIT, 'hf_model_id': HF_ID, 'hf_revision': HF_REVISION}}

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_dir = Path(self.config['model_dir']).resolve()
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
            self._model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
                trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
        return self._tokenizer, self._model

    def predict(self, request_dict, output_dir):
        request = copy.deepcopy(request_dict)
        manifest = self.describe()
        validate_language_request(request, manifest)
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        _write_json(output / 'request.json', request)
        artifacts = [{'name': 'request', 'path': 'request.json', 'kind': 'report', 'media_type': 'application/json'}]
        try:
            import torch
            import transformers
            from transformers import set_seed
            set_seed(request['seed'])
            scorer, scorer_path = load_official_scorer(self.config['source_dir'])
            start = time.monotonic()
            tokenizer, model = self._load()
            formatted = tokenizer.apply_chat_template(request['inputs']['messages'], add_generation_prompt=True, tokenize=False)
            (output / 'prompt.txt').write_text(formatted, encoding='utf-8')
            encoded = tokenizer(formatted, add_special_tokens=False, return_tensors='pt').to(model.device)
            input_tokens = encoded['input_ids'].shape[-1]
            if input_tokens > 7192:
                raise ValueError('Prompt exceeds official 7192-token bound; implicit truncation is disabled')
            max_new_tokens = request.get('parameters', {}).get('max_new_tokens', 4096)
            torch.cuda.reset_peak_memory_stats()
            started_generation = time.monotonic()
            with torch.inference_mode():
                tokens = model.generate(**encoded, do_sample=False, max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            new_tokens = tokens[0, input_tokens:]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
            (output / 'raw_output.txt').write_text(raw, encoding='utf-8')
            artifacts.extend([
                {'name': 'prompt', 'path': 'prompt.txt', 'kind': 'raw_text', 'media_type': 'text/plain'},
                {'name': 'raw_output', 'path': 'raw_output.txt', 'kind': 'raw_text', 'media_type': 'text/plain'}])
            token_list = new_tokens.cpu().tolist()
            _write_json(output / 'generated_token_ids.json', token_list)
            artifacts.append({'name': 'generated_tokens', 'path': 'generated_token_ids.json', 'kind': 'trace', 'media_type': 'application/json'})
            reached_eos = bool(token_list and token_list[-1] == tokenizer.eos_token_id)
            diagnostics = {'parse_status': 'error', 'input_tokens': input_tokens,
                'generated_tokens': len(token_list), 'eos_reached': reached_eos,
                'token_limit_reached': len(token_list) >= max_new_tokens and not reached_eos,
                'generation_seconds': time.monotonic() - started_generation,
                'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
                'device': torch.cuda.get_device_name(), 'decode': 'greedy', 'safety_evaluated': False}
            try:
                partial, parse_details = parse_partial(raw, allow_repair=True)
                diagnostics.update(parse_details)
                _write_json(output / 'partial_state.json', partial)
                recovered = scorer.recover_game_state_from_partial(copy.deepcopy(request['inputs']['current_state']), copy.deepcopy(partial), has_score=True)
                _write_json(output / 'predicted_state.json', recovered)
                artifacts.extend([
                    {'name': 'partial_state', 'path': 'partial_state.json', 'kind': 'symbolic_state', 'media_type': 'application/json', 'semantics': 'native modified/removed/score state difference'},
                    {'name': 'predicted_state', 'path': 'predicted_state.json', 'kind': 'symbolic_state', 'media_type': 'application/json', 'semantics': 'official recovery from current state and generated partial update'}])
            except Exception as exc:
                diagnostics.update(parse_status='error', parse_error=f'{type(exc).__name__}: {exc}')
            diagnostics['total_seconds'] = time.monotonic() - start
            _write_json(output / 'diagnostics.json', diagnostics)
            artifacts.append({'name': 'diagnostics', 'path': 'diagnostics.json', 'kind': 'report', 'media_type': 'application/json'})
            provenance = {**manifest['provenance'], 'scorer_path': str(scorer_path),
                'model_dir': str(self.config['model_dir']),
                'runtime': 'transformers.generate/local checkpoint/bfloat16/SDPA',
                'protocol_deviations': ['Single-case greedy Transformers inference; not distributed verl/vLLM evaluation.',
                    'Schema validation rejects malformed partial updates instead of silently accepting all repairs.']}
            resource_manifest = self.config.get('resource_manifest')
            if resource_manifest:
                resources = json.loads(Path(resource_manifest).read_text(encoding='utf-8'))
                provenance['resources'] = resources
            return write_prediction_bundle(output, request, manifest, artifacts, mode='one_step',
                diagnostics=diagnostics, provenance=provenance,
                claims=['Real author-checkpoint generation; prediction does not establish safety or benchmark performance.'])
        except Exception as exc:
            _write_json(output / 'error.json', {'status': 'error', 'error_type': type(exc).__name__, 'message': str(exc)})
            raise