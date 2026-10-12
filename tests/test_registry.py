"""Recipe ownership, machine-file selection and extension loading contracts."""
import json
from pathlib import Path
import sys

import pytest

from wms.core import registry as module
from wms.core.registry import Registry


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    return tmp_path


def test_default_discovery_external_file_dedup_and_duplicate_id(project):
    first = write(project / "native/new_model/recipes.json", {
        "recipes": {"new": {"task": "native", "entrypoint": "some_model:run"}}})
    external = write(project / "external/recipes.json", {
        "recipes": {"external": {"task": "native", "entrypoint": "other_model:run"}}})
    registry = Registry(paths=[first, external, external])
    assert set(registry.items["recipes"]) == {"new", "external"}
    duplicate = write(project / "duplicate.json", {"recipes": {"new": {"task": "native"}}})
    with pytest.raises(ValueError, match="Duplicate recipes ID 'new'") as caught:
        registry.load(duplicate)
    assert str(first) in str(caught.value) and str(duplicate) in str(caught.value)
    assert registry.items["recipes"]["new"]["entrypoint"] == "some_model:run"


def test_runtime_precedence_and_environment_binding(project):
    folder = project / "native/new_model"
    write(folder / "default.json", {"python": "default-python", "env": {"DEFAULT": "yes"}})
    mapped = write(folder / "server.json", {
        "python": "server-python", "cuda_visible_devices": "3", "env": {"SERVER": "yes"}})
    selected = write(folder / "selected.json", {
        "python": "selected-python", "cuda_visible_devices": "5", "cwd": "work",
        "python_paths": ["lib"], "env": {"FILE": "yes", "PRIORITY": "file"}, "weights": "not-a-runner-field"})
    write(folder / "recipes.json", {"recipes": {"new": {
        "task": "native", "entrypoint": "some_model:run",
        "runtime": {"native_key": "new", "config_file": "default.json", "env": {"RECIPE": "yes", "PRIORITY": "recipe"}},
        "environment": {"entrypoint": "some_env:Env", "config": {"runtime_config": "$runtime.config_file", "env": "$runtime.env"}}
    }}})
    registry = Registry(native_configs={"new": str(mapped)})
    mapped_config = registry.resolve({"recipe": "new"})
    assert mapped_config["runtime"]["config_file"] == str(mapped)
    assert mapped_config["runtime"]["python"] == "server-python"
    config = registry.resolve({"recipe": "new", "runtime": {
        "config_file": str(selected), "python": sys.executable,
        "env": {"CUDA_VISIBLE_DEVICES": "7", "PRIORITY": "request"}}})
    runtime = config["runtime"]
    assert runtime["config_file"] == str(selected)
    assert config["environment"]["config"]["runtime_config"] == str(selected)
    assert runtime["python"] == sys.executable
    assert runtime["env"] == {"CUDA_VISIBLE_DEVICES": "7", "FILE": "yes", "RECIPE": "yes", "PRIORITY": "request"}
    assert config["environment"]["config"]["env"] == runtime["env"]
    assert runtime["cwd"] == str(folder / "work")
    assert runtime["python_paths"] == [str(folder / "lib")]
    assert "weights" not in runtime


def test_missing_machine_config_is_visible_and_can_be_overridden(project):
    write(project / "native/new/recipes.json", {"recipes": {"new": {
        "task": "native", "runtime": {"config_file": "missing.json"}}}})
    registry = Registry()
    listing = registry.recipes()[0]
    assert listing["configured"] is False
    assert "missing.json" in listing["configuration_error"]
    assert registry.recipe("new")["configured"] is False
    with pytest.raises(ValueError, match="Machine configuration not found"):
        registry.resolve({"recipe": "new"})
    machine = write(project / "valid.json", {"python": sys.executable})
    assert registry.resolve({"recipe": "new", "runtime": {"config_file": str(machine)}})["configured"] is True


def test_named_runtime_resolves_relative_to_its_declaration(project, monkeypatch):
    folder = project / "external"
    machine = write(folder / "machine.json", {"python": "venv/python", "env": {"LOCAL": "yes"}})
    config = write(folder / "registry.json", {
        "runtimes": {"work": {"config_file": "machine.json"}},
        "recipes": {"new": {"task": "native", "runtime": "work"}}})
    elsewhere = project / "other"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    run = Registry(paths=[config]).resolve({"recipe": "new"})
    assert run["runtime"]["config_file"] == str(machine)
    assert run["runtime"]["python"] == str(folder / "venv/python")


def test_invalid_machine_json_does_not_hide_other_recipes(project):
    write(project / "native/new/machine.json", [])
    write(project / "native/new/recipes.json", {"recipes": {
        "bad": {"task": "native", "runtime": {"config_file": "machine.json"}},
        "good": {"task": "native", "entrypoint": "some_module:run"}}})
    recipes = {item["id"]: item for item in Registry().recipes()}
    assert recipes["good"]["configured"] is True
    assert recipes["bad"]["configured"] is False
    assert "JSON object" in recipes["bad"]["configuration_error"]


def test_python_paths_do_not_modify_service_import_path(project):
    write(project / "native/custom/recipes.json", {
        "python_paths": ["lib"], "recipes": {"new": {"task": "native"}}})
    before = list(sys.path)
    config = Registry().resolve({"recipe": "new"})
    assert sys.path == before
    assert config["runtime"]["python_paths"] == [str(project / "native/custom/lib")]


def test_explicit_legacy_profile_mapping_selects_existing_machine_file(project):
    default = write(project / "native/new/default.json", {"python": "default-python"})
    profile = write(project / "server/profile.json", {"adapter": "new", "python": "server-python"})
    collection = write(project / "server/runtimes.json", {"profiles": ["profile.json"]})
    write(project / "native/new/recipes.json", {"recipes": {"predict-new": {
        "task": "native", "runtime": {"native_key": "predict-new", "config_file": "default.json"}}}})
    assert Registry().resolve({"recipe": "predict-new"})["runtime"]["config_file"] == str(default)
    actual = Registry(native_configs={"world_model": str(collection)}).resolve({"recipe": "predict-new"})
    assert actual["runtime"]["config_file"] == str(profile)
    assert actual["runtime"]["python"] == "server-python"


def test_builtin_ids_defaults_and_machine_paths_are_still_available():
    registry = Registry()
    expected = {"native-wan", "native-modelscope", "native-dreamer", "native-maniskill", "native-openvla", "native-renew",
                "predict-dreamer", "predict-rlvr_language", "predict-rlvr_video", "maniskill-hold"}
    assert expected <= set(registry.items["recipes"])
    wan = registry.recipe("native-wan")
    assert wan["inputs"]["generation"]["frame_num"] == 81
    assert wan["inputs"]["generation"]["sampling_steps"] == 50
    assert wan["entrypoint"] == "wms.engines.wan.native:wan_generate"
    overridden = registry.resolve({"recipe": "native-wan", "inputs": {"generation": {"frame_num": 17}},
                                   "runtime": {"config_file": None, "python": sys.executable}})
    assert overridden["inputs"]["generation"]["frame_num"] == 17
    assert overridden["inputs"]["generation"]["sampling_steps"] == 50
    assert registry.recipe("native-maniskill")["steps"] == 500
    assert len(registry.recipe("native-renew")["inputs"]["methods"]) == 6


def test_new_prediction_adapter_can_be_loaded_without_editing_shared_map(project, monkeypatch):
    package = project / "new_predictor.py"
    package.write_text('''import json
class Adapter:
    def __init__(self, config): self.config = config
    def describe(self): return {"model_id": "new-model"}
    def predict(self, request, output):
        value = {"model_id": request["model_id"], "outputs": [], "state": request["inputs"]["state"] + 1}
        (output / "prediction.json").write_text(json.dumps(value))
        return value
''', encoding="utf-8")
    monkeypatch.syspath_prepend(str(project))
    from wms.engines.native_tasks import PredictionModel
    model = PredictionModel({"adapter_entrypoint": "new_predictor:Adapter", "runtime_config": {}})
    result = model.predict({"state": 4, "output_dir": str(project)})
    assert result["data"]["state"] == 5
    assert result["outputs"][-1]["path"] == "prediction.json"
