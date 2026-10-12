"""Build a CPU-only, immutable-evidence report for the center-precision study."""
from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path, PurePosixPath
import re
from statistics import fmean
from urllib.parse import quote

TASK_ID = "WMS-PushCubeCenter-v1"
REWARD_VERSION = "center-dense/1"


def _escape(value):
    return html.escape(str(value), quote=True)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _source(output, run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"run_[0-9a-f]+", run_id):
        raise ValueError("Use an existing platform run ID")
    path = output.parents[1] / run_id / "output"
    if path.resolve() == output.resolve():
        raise ValueError("The report must have its own output directory")
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _url(run_id, relative):
    path = PurePosixPath(str(relative).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts or ":" in str(path):
        raise ValueError("Artifact links must stay inside the source output directory")
    return f"/api/v1/runs/{run_id}/files/" + quote(path.as_posix(), safe="/")


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must contain a finite measured number")
    return value


def _flag(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must contain a measured boolean")
    return value


def _aggregate(report):
    if report.get("env_id") != TASK_ID:
        raise ValueError("Center study report requires the independent center task")
    if report.get("protocol", {}).get("reward_version") != REWARD_VERSION:
        raise ValueError("Center study report requires matching reward semantics")
    episodes = report.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("A completed evaluation with all episode outcomes is required")
    distances, centers, stable, regions = [], [], [], []
    for ep in episodes:
        distance = _number(ep.get("final_goal_distance_m"), "final_goal_distance_m")
        if distance < 0:
            raise ValueError("A distance cannot be negative")
        distances.append(distance)
        centers.append(_flag(ep.get("center_success_at_end"), "center_success_at_end"))
        stable.append(_flag(ep.get("center_stable_last_5"), "center_stable_last_5"))
        regions.append(_flag(ep.get("region_success_at_end"), "region_success_at_end"))
    seeds = [ep["seed"] for ep in episodes]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Repeated evaluation seeds would duplicate evidence")
    return {"episodes": len(episodes), "center_successes": sum(centers),
            "center_success_rate": fmean(centers), "center_stable_successes": sum(stable),
            "center_stable_last_5_rate": fmean(stable), "region_successes": sum(regions),
            "region_success_rate": fmean(regions), "final_distance_mean_m": fmean(distances),
            "final_distance_max_m": max(distances), "seeds": seeds,
            "failed_center_seeds": [ep["seed"] for ep in episodes if not ep["center_success_at_end"]]}


def _checkpoint_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _evaluation(path, training, training_source, checkpoint_hashes, *, initial=False):
    report = _read(path / "evaluation.json")
    if report.get("task_sha256") != training.get("task_sha256"):
        raise ValueError("Training and evaluation task source identities differ")
    value = report.get("checkpoint")
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError("Evaluation needs the absolute path of its training checkpoint")
    checkpoint = Path(value).resolve()
    if checkpoint.parent != training_source.resolve():
        raise ValueError("Evaluation checkpoint must come from the selected training_run output")
    if initial:
        if checkpoint.name != "initial.pt":
            raise ValueError("Initial evaluation must use the selected training initial.pt")
    else:
        allowed = {"model.pt", *training.get("checkpoints", [])} - {"initial.pt"}
        if checkpoint.name not in allowed:
            raise ValueError("Evaluation must use model.pt or a checkpoint listed by the selected training")
    if checkpoint not in checkpoint_hashes:
        checkpoint_hashes[checkpoint] = _checkpoint_sha256(checkpoint)
    if report.get("checkpoint_sha256") != checkpoint_hashes[checkpoint]:
        raise ValueError("Evaluation checkpoint_sha256 differs from the actual training checkpoint")
    return report, _aggregate(report)


def _link(run_id, name, label):
    return f'<a href="{_escape(_url(run_id, name))}" target="_blank" rel="noreferrer">{_escape(label)}</a>'


def _distance(value):
    return "未记录" if value is None else f"{float(value) * 100:.2f} cm"


def _yes(value):
    return "是" if value is True else "否" if value is False else "未记录"


def _media(run_id, source, record):
    sources, links, missing = [], [], []
    for field, media, label in (("video_webm", "video/webm", "WebM"), ("video", "video/mp4", "MP4")):
        filename = record.get(field)
        if not filename:
            continue
        url = _url(run_id, filename)
        if not (source / filename).is_file():
            missing.append(str(filename))
            continue
        sources.append(f'<source src="{_escape(url)}" type="{media}">')
        links.append(_link(run_id, filename, label))
    poster = record.get("video_poster")
    poster_exists = bool(poster and (source / poster).is_file())
    poster_attr = f' poster="{_escape(_url(run_id, poster))}"' if poster_exists else ""
    value = ('<video controls preload="none" playsinline' + poster_attr + '>' + ''.join(sources)
             + '当前播放器不支持此视频，请使用下方链接。</video>') if sources else '<p class="muted">本回合没有可用的录像文件。</p>'
    if record.get("video_duration_s") is not None:
        value += f'<p class="muted">实际回放 {_escape(format(float(record["video_duration_s"]), ".2f"))} 秒；{_escape(record.get("video_frames", "未记录"))} 帧 / {_escape(record.get("video_fps", "未记录"))} fps</p>'
    if links:
        value += '<p class="media-links">' + ' · '.join(links) + '</p>'
    if poster_exists:
        value += '<details><summary>查看最后一帧</summary><img loading="lazy" src="' + _escape(_url(run_id, poster)) + '" alt="该回合最后一帧真实仿真画面"></details>'
    if missing:
        value += '<p class="warning">记录引用的文件不存在：' + _escape(', '.join(missing)) + '</p>'
    return value


def _episode_cards(run_id, source, report):
    cards = []
    for ep in report["episodes"]:
        ok = ep["center_success_at_end"]
        title = f'种子 {ep["seed"]} · ' + ('中心任务成功' if ok else '中心任务失败')
        values = [('最终圆心误差', _distance(ep["final_goal_distance_m"])),
                  ('最后5步稳定', _yes(ep["center_stable_last_5"])),
                  ('原10cm标准参考', _yes(ep["region_success_at_end"])),
                  ('实际动作步数', ep.get("steps", "未记录"))]
        cards.append('<article class="card"><h3>' + _escape(title) + '</h3><dl>'
                     + ''.join(f'<div><dt>{_escape(k)}</dt><dd>{_escape(v)}</dd></div>' for k, v in values)
                     + '</dl>' + _media(run_id, source, ep) + '</article>')
    return '<div class="video-grid">' + ''.join(cards) + '</div>'


def _training_cards(run_id, source, recordings):
    cards = []
    for record in recordings:
        title = f'训练步 {record["train_step_start"]:,} → {record["train_step_end"]:,}'
        cards.append('<article class="card"><h3>' + _escape(title) + '</h3><p class="muted">环境0的真实训练采集 · 探索策略，非固定检查点评测</p><dl>'
                     + '<div><dt>该回合回报</dt><dd>' + _escape(f'{record["return"]:.3f}') + '</dd></div>'
                     + '<div><dt>最终任务成功</dt><dd>' + _yes(record.get("success_at_end")) + '</dd></div>'
                     + '<div><dt>最终圆心误差</dt><dd>' + _distance(record.get("final_goal_distance_m")) + '</dd></div>'
                     + '<div><dt>随机初始化动作步数</dt><dd>' + _escape(record.get("random_action_steps", "未记录")) + '</dd></div></dl>'
                     + _media(run_id, source, record) + '</article>')
    return '<div class="video-grid">' + ''.join(cards) + '</div>'


def _curve(history, field, title, rate=False):
    points = [(_number(row["environment_steps"], "environment_steps"), _number(row[field], field))
              for row in history if field in row]
    if not points:
        return '<p class="muted">' + _escape(title) + '：没有记录。</p>'
    xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
    ymin, ymax = (0, 1) if rate else (min(y for _, y in points), max(y for _, y in points))
    if ymin == ymax:
        ymin, ymax = ymin - .5, ymax + .5
    width, height, left, right, top, bottom = 640, 260, 56, 18, 32, 44
    def xy(x, y):
        return (left + (x - xmin) / max(1, xmax - xmin) * (width - left - right),
                top + (ymax - y) / (ymax - ymin) * (height - top - bottom))
    line = ' '.join(f'{x:.2f},{y:.2f}' for x, y in (xy(x, y) for x, y in points))
    labels = [(left - 8, top + 5, f'{ymax:.3f}', 'end'),
              (left - 8, height - bottom + 5, f'{ymin:.3f}', 'end'),
              (left, height - 17, f'{xmin:,.0f}', 'start'),
              (width - right, height - 17, f'{xmax:,.0f}', 'end')]
    return ('<figure><figcaption>' + _escape(title) + '</figcaption>'
            + f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}，横轴为累计训练环境步数">'
            + f'<path d="M {left} {top} V {height-bottom} H {width-right}" fill="none" stroke="#94a3b8"/>'
            + f'<polyline points="{line}" fill="none" stroke="#2563eb" stroke-width="2"/>'
            + ''.join(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="12" fill="#526175">{_escape(label)}</text>' for x,y,label,anchor in labels)
            + '</svg><p class="muted">训练采集统计；不是独立测试成功率。横轴：累计环境步数。</p></figure>')


def _summary_row(label, summary):
    return '<tr>' + ''.join('<td>' + _escape(value) + '</td>' for value in (
        label, f'{summary["center_successes"]}/{summary["episodes"]}',
        _distance(summary["final_distance_mean_m"]),
        f'{summary["center_stable_successes"]}/{summary["episodes"]}',
        f'{summary["region_successes"]}/{summary["episodes"]}')) + '</tr>'


def build(config, output_dir, context):
    """Read existing run evidence and create a separate report; no model calls."""
    output = Path(output_dir).resolve()
    inputs = dict(config.get("inputs", {}))
    training_id, test_id = inputs["training_run"], inputs["test_run"]
    training_source, test_source = _source(output, training_id), _source(output, test_id)
    training = _read(training_source / "training.json")
    if training.get("env_id") != TASK_ID or training.get("reward_version") != REWARD_VERSION:
        raise ValueError("Select training for WMS-PushCubeCenter-v1 / center-dense/1")
    history = [json.loads(line) for line in (training_source / "training_history.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    recording_manifest = _read(training_source / "training_recordings.json")
    recordings = recording_manifest["recordings"]
    checkpoint_hashes = {}
    test, test_summary = _evaluation(test_source, training, training_source, checkpoint_hashes)
    initial_id = inputs.get("initial_run")
    initial = initial_summary = initial_source = None
    if initial_id:
        initial_source = _source(output, initial_id)
        initial, initial_summary = _evaluation(initial_source, training, training_source, checkpoint_hashes, initial=True)
    developments = []
    for run_id in inputs.get("development_runs", []):
        context.check_cancelled()
        source = _source(output, run_id)
        report, summary = _evaluation(source, training, training_source, checkpoint_hashes)
        developments.append({"run_id": run_id, "summary": summary, "checkpoint": report.get("checkpoint")})
    development_seeds = {seed for item in developments for seed in item["summary"]["seeds"]}
    overlap = sorted(development_seeds.intersection(test_summary["seeds"]))
    summary = {"task": "center_experiment_report", "env_id": TASK_ID, "reward_version": REWARD_VERSION,
               "source_runs": {"training": training_id, "test": test_id, "initial": initial_id,
                               "development": [item["run_id"] for item in developments]},
               "training": training, "test": test_summary, "initial": initial_summary,
               "developments": developments, "development_test_seed_overlap": overlap,
               "training_recordings": recordings, "test_episodes": test["episodes"],
               "task_sha256": training.get("task_sha256"),
               "test_checkpoint_sha256": test.get("checkpoint_sha256"),
               "safety_status": "not_measured"}
    cards = [f'<div class="stat"><span>最终中心成功（≤2cm且在桌面）</span><strong>{test_summary["center_successes"]}/{test_summary["episodes"]}</strong><small>{test_summary["center_success_rate"]:.1%}</small></div>',
             f'<div class="stat"><span>最终圆心平均误差</span><strong>{_distance(test_summary["final_distance_mean_m"])}</strong><small>所有测试回合，含失败</small></div>',
             f'<div class="stat"><span>最后5步均在中心范围</span><strong>{test_summary["center_stable_successes"]}/{test_summary["episodes"]}</strong><small>{test_summary["center_stable_last_5_rate"]:.1%}</small></div>',
             f'<div class="stat"><span>原10cm区域标准参考</span><strong>{test_summary["region_successes"]}/{test_summary["episodes"]}</strong><small>单独保留，不能代替中心任务</small></div>']
    text = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TD-MPC2 精准推到圆心 · 实验记录</title><style>
:root{font:16px/1.6 system-ui,-apple-system,"Microsoft YaHei",sans-serif;color:#182538;background:#f3f6fb}*{box-sizing:border-box}body{margin:0}main{max-width:1220px;margin:auto;padding:28px 24px 64px}h1{font-size:clamp(26px,4vw,40px);line-height:1.25;margin:0 0 16px}h2{font-size:25px;margin:40px 0 14px}h3{font-size:18px;margin:0 0 10px}a{color:#155bce;overflow-wrap:anywhere}.eyebrow{letter-spacing:.08em;font-size:13px;color:#526175;font-weight:700}.intro,.notice{background:white;padding:20px 24px;border-radius:12px;border:1px solid #dce5f0}.notice{border-left:5px solid #326fe0}.muted{color:#526175;font-size:14px}.warning{background:#fff1d5;padding:12px;border-radius:8px;color:#725008}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:22px 0}.stat{display:flex;flex-direction:column;padding:20px;background:white;border:1px solid #dce5f0;border-radius:12px}.stat span,.stat small{font-size:13px;color:#526175}.stat strong{font-size:30px;color:#174bb4;padding:6px 0}.video-grid,.curves{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.card,figure{background:white;border:1px solid #dce5f0;border-radius:12px;padding:20px;margin:0}.card dl{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:14px 0}.card dt{color:#526175;font-size:13px}.card dd{margin:0;font-weight:650}video,img{width:100%;height:auto;display:block;border-radius:8px;background:#e3eaf2}video{max-height:560px}details{margin-top:12px}summary{cursor:pointer;color:#155bce;margin-bottom:10px}figcaption{font-weight:650}svg{width:100%;height:auto;display:block}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;background:white}td,th{padding:12px;border-bottom:1px solid #dce5f0;text-align:left}th{font-weight:650;background:#eaf0fa}code{font-size:13px;overflow-wrap:anywhere}.source-list{overflow-wrap:anywhere}.source-list li{margin:8px 0}@media(max-width:900px){.stats{grid-template-columns:1fr 1fr}}@media(max-width:660px){main{padding:20px 14px 40px}.video-grid,.curves{grid-template-columns:1fr}.card{padding:16px}.intro,.notice{padding:16px}.stat{padding:14px}.stat strong{font-size:26px}}
</style></head><body><main><p class="eyebrow">WM-SAFETYLAB · TD-MPC2 × MANISKILL</p><h1>精准推到圆心：训练过程与完整测试</h1>'''
    text += '<div class="intro"><p><strong>独立任务 WMS-PushCubeCenter-v1</strong>：保留原始20cm初始推动位移，采用新的连续圆心奖励与2cm成功容差；原 PushCube-v1 任务及历史结果保持原定义。</p>'
    text += '<p>成功要求：最终 XY 距离 ≤2cm，且方块中心高度距离桌面正常高度2cm不超过5mm。最后5步稳定率要求最后5个动作后的状态全部满足该条件；曾经经过圆心不算最终成功。</p>'
    text += '<p>每回合50步，控制频率20Hz。MPPI 使用模型内部潜在动力学和预测回报选择动作；此报告只读取已有证据。</p></div><div class="stats">' + ''.join(cards) + '</div>'
    text += '<div class="notice">本次为单个训练随机种子的工程研究实验。任务完成率不等于安全性；碰撞、接触力等安全指标尚未测量。原始任务与新任务的奖励定义不同，不能跨任务直接比较回报。</div>'
    if overlap:
        text += '<p class="warning">测试与开发评测使用了重复种子：' + _escape(overlap) + '。这些结果不能声明为完全独立的保留测试集。</p>'
    text += '<h2>1. 实验来源与运行范围</h2><ul class="source-list">'
    for label, run_id, filename in [('训练记录', training_id, 'training.json'), ('完整测试报告', test_id, 'visualization.html'), ('测试原始数据', test_id, 'evaluation.json')]:
        text += '<li>' + _link(run_id, filename, label) + ' · <code>' + _escape(run_id) + '</code></li>'
    text += '</ul><p>训练随机种子：<strong>' + _escape(training.get("seed", "未记录")) + '</strong>；训练环境步数：<strong>' + _escape(training.get("environment_steps", "未记录")) + '</strong>；梯度更新次数：<strong>' + _escape(training.get("update_count", "未记录")) + '</strong>。</p>'
    text += '<p class="muted">测试检查点：<code>' + _escape(test.get("checkpoint", "未记录")) + '</code><br>SHA-256：<code>' + _escape(test.get("checkpoint_sha256", "未记录")) + '</code></p>'
    text += '<p>测试种子：<code>' + _escape(test_summary["seeds"]) + '</code>。所有回合参与汇总，失败种子：<code>' + _escape(test_summary["failed_center_seeds"]) + '</code>。</p>'
    if initial_summary:
        text += '<h2>2. 初始与训练后检查点对照</h2><p>两列结果均采用独立中心任务的同一成功定义。</p>'
        if initial_summary["seeds"] != test_summary["seeds"]:
            text += '<p class="warning">初始与训练后使用的种子列表不同；这里是各自完整样本的汇总，不是逐种子的配对比较。</p>'
        text += '<div class="table-wrap"><table><tr><th>检查点</th><th>2cm最终成功</th><th>最终平均误差</th><th>末5步稳定</th><th>原10cm参考</th></tr>' + _summary_row('初始检查点', initial_summary) + _summary_row('训练后测试检查点', test_summary) + '</table></div>'
        text += '<p>' + _link(initial_id, 'visualization.html', '初始检查点完整评测及录像') + '</p>'
    text += '<h2>训练曲线</h2><div class="curves">' + _curve(history, 'collection_return', '训练采集平均回报') + _curve(history, 'collection_success_once', '训练采集曾成功比例', rate=True) + '</div>'
    text += '<h2>真实训练过程录像</h2><p>共记录 ' + str(len(recordings)) + ' 段预定训练回合，覆盖起始、约25%、50%、75%和最后一批。画面来自训练中的环境0，包含探索动作；未额外重置、未替换动作，也没有筛选最好的片段。一个片段期间模型可能继续更新，不能视为固定权重的独立测试。</p>'
    text += '<p>' + _link(training_id, 'training_recordings.json', '训练录像逐段元数据') + ' · ' + _link(training_id, 'training_history.jsonl', '完整训练曲线数据') + '</p>'
    text += _training_cards(training_id, training_source, recordings)
    if developments:
        text += '<h2>开发评测记录</h2><p>这些评测用于检查训练进展，单独列出，不计入上方测试汇总。</p><div class="table-wrap"><table><tr><th>运行</th><th>2cm最终成功</th><th>最终平均误差</th><th>末5步稳定</th><th>原10cm参考</th></tr>'
        for item in developments:
            text += _summary_row(item["run_id"], item["summary"])
        text += '</table></div><p>' + ' · '.join(_link(item["run_id"], 'visualization.html', item["run_id"]) for item in developments) + '</p>'
    text += '<h2>全部测试回合录像</h2><p>按原始测试顺序展示全部 ' + str(test_summary["episodes"]) + ' 个回合，包括失败；没有按效果挑选或排序。播放器优先尝试 WebM，再尝试 MP4；也可直接打开格式链接或展开查看末帧。</p>'
    text += _episode_cards(test_id, test_source, test)
    text += '<h2>结论边界与复查</h2><p>2cm是可测量的成功容差，不表示数学意义的零误差。末5步稳定是额外位置与桌面状态检查，不代表力学稳定性或完整安全验证。增加训练随机种子、更广初态及扰动测试后，才能判断鲁棒性。</p>'
    text += '<p>模型采用状态输入；录像是实际仿真画面，不是模型生成的视频。所有统计来自原始训练和评测JSON文件，本报告没有重新执行模型或改写原始证据。</p><p><a href="evidence_summary.json">下载汇总证据 JSON</a></p></main></body></html>'
    output.mkdir(parents=True, exist_ok=True)
    context.write_json("evidence_summary.json", summary)
    (output / "experiment.html").write_text(text, encoding="utf-8")
    from .native import _result
    return _result(output, summary)
