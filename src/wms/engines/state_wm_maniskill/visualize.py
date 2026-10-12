"""Render evidence-aligned ManiSkill videos and numerical prediction comparisons.

The curves are retrospective rolling forecasts, not a model-driven controller.
At video frame k, an h-step forecast was initialized at the real state k-h and
rolled forward using the recorded actions. Missing early forecasts stay missing.
"""
from pathlib import Path
import csv
import json


METHODS = ("action_conditioned", "no_action", "persistence")
LABELS = {"action_conditioned": "Action-conditioned WM", "no_action": "No-action WM",
          "persistence": "State persistence"}
COLORS = {"truth": "#263442", "action_conditioned": "#0072B2",
          "no_action": "#D55E00", "persistence": "#009E73"}
STYLES = {"truth": "-", "action_conditioned": "-", "no_action": "--", "persistence": ":"}


def align_episode(data, predictions, episode_id, seed_index, horizon, fps):
    """Return frame-aligned truth and forecasts; leading unavailable rows are NaN."""
    import numpy as np

    if horizon < 1 or fps <= 0:
        raise ValueError("A positive horizon and frame rate are required")
    ids = np.flatnonzero(data["episode_ids"] == episode_id)
    if not len(ids) or not np.array_equal(data["steps"][ids], np.arange(len(ids))):
        raise ValueError(f"Episode {episode_id} must contain consecutive steps starting at zero")
    if len(ids) > 1 and not np.allclose(data["next_states"][ids[:-1]], data["states"][ids[1:]], atol=1e-6):
        raise ValueError("Episode transitions are not contiguous")
    truth = np.concatenate([data["states"][ids[:1]], data["next_states"][ids]], axis=0)
    if not np.isfinite(truth).all():
        raise ValueError("Non-finite truth data")
    if horizon == 1:
        selected = np.flatnonzero(predictions["episode_ids"] == episode_id)
        starts = predictions["steps"][selected].astype(int)
        reference = predictions["truth"][selected]
        initial = predictions["initial"][selected]
        values = {m: predictions[f"{m}_one_step"][seed_index, selected] for m in METHODS[:2]}
    else:
        if horizon > predictions["rollout_truth"].shape[1]:
            raise ValueError("Requested horizon exceeds saved rollout predictions")
        selected = np.flatnonzero(predictions["rollout_episode_ids"] == episode_id)
        starts = predictions["rollout_steps"][selected].astype(int)
        reference = predictions["rollout_truth"][selected, horizon - 1]
        initial = predictions["rollout_initial"][selected]
        values = {m: predictions[f"{m}_rollout"][seed_index, selected, horizon - 1] for m in METHODS[:2]}
    targets = starts + horizon
    if (not len(targets) or len(np.unique(targets)) != len(targets)
            or (starts < 0).any() or (targets >= len(truth)).any()):
        raise ValueError("Missing, duplicated or out-of-range prediction target frames")
    if not np.allclose(reference, truth[targets], atol=1e-6):
        raise ValueError("Saved prediction truth disagrees with the video frame state")
    if not np.allclose(initial, truth[starts], atol=1e-6):
        raise ValueError("Saved prediction initial state disagrees with the rollout start")
    values["persistence"] = initial
    forecasts, errors = {}, {}
    for method, value in values.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite predictions: {method}")
        aligned = np.full(truth.shape, np.nan, dtype=np.float64)
        aligned[targets] = value
        forecasts[method] = aligned
        errors[method] = {name: np.linalg.norm((aligned[:, part] - truth[:, part]) * 1000.0, axis=1)
                          for name, part in (("cube", slice(18, 21)), ("tcp", slice(21, 24)))}
    return {"episode": int(episode_id), "horizon": int(horizon), "time_s": np.arange(len(truth)) / fps,
            "truth": truth, "forecasts": forecasts, "errors_mm": errors,
            "valid_frames": targets, "start_frames": starts}


def _write_csv(path, aligned, seed, fps):
    import numpy as np
    axes = ("x", "y", "z")
    fields = ["episode", "horizon_steps", "training_seed", "method", "frame", "time_s",
              "forecast_start_frame", "forecast_start_time_s", "prediction_available"]
    fields += [f"{kind}_{entity}_{axis}_mm" for kind in ("truth", "prediction")
               for entity in ("cube", "tcp") for axis in axes]
    fields += ["cube_error_mm", "tcp_error_mm"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for values in aligned:
            starts = dict(zip(values["valid_frames"].tolist(), values["start_frames"].tolist()))
            for frame, moment in enumerate(values["time_s"]):
                for method in METHODS:
                    available = frame in starts
                    row = {"episode": values["episode"], "horizon_steps": values["horizon"],
                           "training_seed": "" if method == "persistence" else seed, "method": method,
                           "frame": frame, "time_s": moment, "forecast_start_frame": starts.get(frame, ""),
                           "forecast_start_time_s": starts[frame] / fps if available else "",
                           "prediction_available": int(available)}
                    for kind, array in (("truth", values["truth"]), ("prediction", values["forecasts"][method])):
                        for entity, start in (("cube", 18), ("tcp", 21)):
                            for axis, offset in zip(axes, range(3)):
                                value = float(array[frame, start + offset]) * 1000.0
                                row[f"{kind}_{entity}_{axis}_mm"] = value if np.isfinite(value) else ""
                    for entity in ("cube", "tcp"):
                        value = float(values["errors_mm"][method][entity][frame])
                        row[f"{entity}_error_mm"] = value if np.isfinite(value) else ""
                    writer.writerow(row)


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                         "axes.titleweight": "bold", "axes.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#B9C4CD", "axes.labelcolor": "#425467",
                         "xtick.color": "#526172", "ytick.color": "#526172",
                         "axes.grid": True, "grid.alpha": 0.18, "grid.color": "#8699AA",
                         "legend.frameon": False, "lines.linewidth": 1.7,
                         "figure.facecolor": "#F4F7FA", "axes.facecolor": "white",
                         "savefig.facecolor": "#F4F7FA", "pdf.fonttype": 42})
    return plt


def _legend(fig, y):
    from matplotlib.lines import Line2D
    lines = [Line2D([], [], color=COLORS[key], linestyle=STYLES[key], linewidth=2.6,
                    label="Real simulator state" if key == "truth" else LABELS[key])
             for key in ("truth", *METHODS)]
    fig.legend(handles=lines, ncol=4, loc="upper left", bbox_to_anchor=(0.035, y),
               borderaxespad=0, fontsize=11, columnspacing=2.5)


def _position_axes(fig, values, positions):
    """Create six component plots, returning axes for the moving video cursors."""
    result = []
    for entity, start, row in (("Cube", 18, 0), ("End effector (TCP)", 21, 1)):
        for coordinate, axis in enumerate(("x", "y", "z")):
            ax = fig.add_axes(positions[row][coordinate])
            for method in ("truth", *METHODS):
                state = values["truth"] if method == "truth" else values["forecasts"][method]
                ax.plot(values["time_s"], state[:, start + coordinate] * 1000,
                        color=COLORS[method], linestyle=STYLES[method], linewidth=1.5 if method == "truth" else 1.6)
            ax.set_title(f"{entity} / {axis.upper()}", loc="left", pad=7)
            if coordinate == 0:
                ax.set_ylabel("Position (mm)")
            ax.set_xlim(0, values["time_s"][-1])
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.locator_params(axis="both", nbins=4)
            if row == 1:
                ax.set_xlabel("Target time (s)")
            result.append(ax)
    return result


def _error_axes(fig, values, positions):
    result = []
    for entity, rectangle in zip(("cube", "tcp"), positions):
        ax = fig.add_axes(rectangle)
        for method in METHODS:
            ax.plot(values["time_s"], values["errors_mm"][method][entity], color=COLORS[method],
                    linestyle=STYLES[method], linewidth=1.8)
        ax.set_title("Cube position error" if entity == "cube" else "End-effector position error", loc="left")
        ax.set_ylabel("Euclidean error (mm)")
        ax.set_xlabel("Target time (s)")
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, values["time_s"][-1])
        ax.locator_params(axis="both", nbins=5)
        result.append(ax)
    return result


def _static_chart(output, values, seed):
    plt = _style()
    fig = plt.figure(figsize=(13.6, 9.0))
    h = values["horizon"]
    fig.text(0.045, .965, f"PushCube | episode {values['episode']:03d} | {h}-step rolling forecasts", size=17, weight="bold", color="#183047")
    fig.text(.045, .931, f"Training seed {seed}; positions in world coordinates; fixed controller, offline WM prediction", size=10, color="#526172")
    _legend(fig, .902)
    positions = [[(.07 + col * .325, .625 - row * .25, .255, .16) for col in range(3)] for row in range(2)]
    _position_axes(fig, values, positions)
    _error_axes(fig, values, [(0.07, .095, .41, .18), (.57, .095, .41, .18)])
    fig.text(.045, .018, f"Forecast at target k starts from real state k-{h}; recorded actions drive recursive prediction. No forecast in the first {h} frames.", size=9, color="#526172")
    stem = f"episode_{values['episode']:03d}_h{h}"
    paths = []
    for extension in ("png", "pdf"):
        path = output / "charts" / f"{stem}.{extension}"
        fig.savefig(path, dpi=300 if extension == "png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def _comparison_figure(values, seed, first_frame, fps, noise_std=0.003):
    import numpy as np
    plt = _style()
    fig = plt.figure(figsize=(16, 10), dpi=100)
    fig.text(.035, .965, "MANISKILL  /  WORLD-MODEL CONSEQUENCE CHECK", size=22, weight="bold", color="#173047")
    subtitle = fig.text(.035, .928, "", size=12, color="#526172")
    _legend(fig, .90)
    camera = fig.add_axes([.035, .375, .335, .45])
    camera.set_axis_off()
    artist = camera.imshow(first_frame)
    camera.set_title("REAL SIMULATION  |  Panda + PushCube", size=12, loc="left", pad=10)
    fig.text(.035, .345, f"Controller: fixed official planner + {noise_std:g} rad noise", size=10, color="#526172")
    fig.text(.035, .32, "WM predicts states offline; it does not control the robot.", size=10, color="#526172")
    status = fig.text(.035, .282, "", size=11, family="DejaVu Sans Mono", color="#173047", linespacing=1.6, va="top")
    positions = [[(.43 + col * .19, .68 - row * .24, .155, .145) for col in range(3)] for row in range(2)]
    axes = _position_axes(fig, values, positions)
    axes += _error_axes(fig, values, [(.43, .125, .24, .18), (.745, .125, .235, .18)])
    cursors = [ax.axvline(0, color="#697A89", linewidth=1, alpha=.8) for ax in axes]
    fig.text(.035, .083, f"Playback 1x  |  {fps:g} control frames / s", size=10, color="#526172")
    fig.text(.035, .048, "Curves show the complete held-out episode; the vertical cursor is the current video frame.", size=10, color="#526172")
    fig.text(.035, .024, f"Each {values['horizon']}-step endpoint starts from the real state {values['horizon']} frames earlier. No forecast is imputed at the beginning.", size=10, color="#526172")

    def update(frame, image):
        artist.set_data(image)
        moment = values["time_s"][frame]
        subtitle.set_text(f"Episode {values['episode']:03d}  |  frame {frame:03d}  |  t = {moment:.2f} s  |  {values['horizon']}-step ({values['horizon']/fps:.2f} s) forecast  |  training seed {seed}")
        for cursor in cursors:
            cursor.set_xdata([moment, moment])
        text = ["CURRENT ERROR (mm)     CUBE     TCP"]
        for method, label in zip(METHODS, ("Action-conditioned", "No-action WM", "Persistence")):
            errors = [values["errors_mm"][method][name][frame] for name in ("cube", "tcp")]
            numbers = [f"{value:7.3f}" if np.isfinite(value) else "     --" for value in errors]
            text.append(f"{label:<20}{numbers[0]} {numbers[1]}")
        status.set_text("\n".join(text))
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

    return fig, update


def _simulation_frame(frame, episode, moment, seed, fps):
    """Letterbox to 960x720 while preserving the camera aspect ratio."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    canvas = Image.new("RGB", (960, 720), "#F4F7FA")
    image = Image.fromarray(frame).convert("RGB")
    image.thumbnail((920, 570), Image.Resampling.LANCZOS)
    canvas.paste(image, ((960 - image.width) // 2, 95 + (570 - image.height) // 2))
    try:
        import matplotlib.font_manager as fonts
        fontpath = fonts.findfont("DejaVu Sans")
        titlefont, bodyfont = ImageFont.truetype(fontpath, 27), ImageFont.truetype(fontpath, 18)
    except (OSError, ImportError):
        titlefont = bodyfont = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 18), "PANDA / PUSHCUBE - REAL SIMULATION", font=titlefont, fill="#173047")
    draw.text((30, 61), f"Held-out episode {episode:03d}  |  simulation seed {seed}  |  t={moment:.2f}s  |  playback 1x", font=bodyfont, fill="#526172")
    draw.text((30, 678), "Fixed motion planner with joint-target noise; world model is evaluated offline.", font=bodyfont, fill="#526172")
    return np.asarray(canvas)


def build_visualizations(output_dir, params, context):
    """Create videos, frame mapping, per-horizon CSVs and per-episode PNG/PDFs."""
    import numpy as np
    import imageio.v2 as imageio

    output = Path(output_dir)
    recordings = json.loads((output / "recordings.json").read_text(encoding="utf-8"))
    metadata = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    training = json.loads((output / "training.json").read_text(encoding="utf-8"))
    data = np.load(output / "dataset.npz", allow_pickle=False)
    predictions = np.load(output / "predictions.npz", allow_pickle=False)
    fps = float(recordings["fps"])
    if not np.isclose(fps, metadata["control_freq_hz"]):
        raise ValueError("Recording FPS must equal the control frequency for frame-aligned playback")
    seed = int(params.get("visualization_seed", 0))
    if seed not in training["seeds"]:
        raise ValueError("Selected visualization seed is not a trained seed")
    seed_index = training["seeds"].index(seed)
    horizon = int(training["horizon"])
    episodes = sorted(recordings["episodes"], key=lambda row: row["episode"])
    if [row["episode"] for row in episodes] != sorted(metadata["split"]["test"]):
        raise ValueError("Visualization must include exactly all held-out test episodes")
    (output / "charts").mkdir(exist_ok=True)
    paths = []
    aligned = {}
    for h in sorted(set((1, horizon))):
        values = [align_episode(data, predictions, row["episode"], seed_index, h, fps) for row in episodes]
        aligned[h] = values
        csv_path = output / f"predictions_h{h}_seed{seed}.csv"
        _write_csv(csv_path, values, seed, fps)
        paths.append(csv_path)
        for values_row in values:
            context.check_cancelled()
            paths.extend(_static_chart(output, values_row, seed))
    simulation_path, comparison_path = output / "simulation.mp4", output / "comparison.mp4"
    video_options = {"fps": fps, "codec": "libx264", "quality": 8, "macro_block_size": 2,
                     "ffmpeg_params": ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]}
    mappings, frame_offset, preview_saved = [], 0, False
    plt = _style()
    with imageio.get_writer(simulation_path, **video_options) as sim_writer, imageio.get_writer(comparison_path, **video_options) as comparison_writer:
        for row, values in zip(episodes, aligned[horizon]):
            episode = int(row["episode"])
            expected = len(values["truth"])
            if int(row["frames"]) != expected or int(row["steps"]) != expected - 1:
                raise ValueError(f"Episode {episode} recording count does not match transitions")
            if row.get("path") is None:
                raise ValueError("A recording file path is required")
            reader = imageio.get_reader(output / row["path"])
            fig, update, actual_frames = None, None, 0
            try:
                for frame, image in enumerate(reader):
                    context.check_cancelled()
                    if frame >= expected:
                        raise ValueError("Video contains more frames than the saved state sequence")
                    if fig is None:
                        fig, update = _comparison_figure(values, seed, image, fps, metadata["action_noise_std_rad"])
                    sim_writer.append_data(_simulation_frame(image, episode, frame / fps, row["seed"], fps))
                    rendered = update(frame, image)
                    comparison_writer.append_data(rendered)
                    if not preview_saved and frame == expected // 2:
                        imageio.imwrite(output / "comparison_preview.png", rendered)
                        paths.append(output / "comparison_preview.png")
                        preview_saved = True
                    actual_frames += 1
            finally:
                reader.close()
                if fig is not None:
                    plt.close(fig)
            if actual_frames != expected:
                raise ValueError(f"Episode {episode}: decoded {actual_frames} frames but expected {expected}")
            mappings.append({"episode": episode, "simulation_seed": row["seed"], "source_video": row["path"],
                             "frames": actual_frames, "montage_start_frame": frame_offset,
                             "montage_end_frame_exclusive": frame_offset + actual_frames,
                             "start_time_s": frame_offset / fps, "end_time_s": (frame_offset + actual_frames) / fps,
                             "forecast_start_frames": values["start_frames"].tolist(),
                             "forecast_target_frames": values["valid_frames"].tolist()})
            frame_offset += actual_frames
            context.event("progress", {"phase": "visualization", "episode": episode,
                                       "completed_episodes": len(mappings), "episodes": len(episodes)})
    paths.extend((simulation_path, comparison_path))
    report = {"schema": "wms.state-wm-maniskill.visualization/1", "fps": fps, "playback_speed": 1,
              "total_frames": frame_offset, "duration_seconds": frame_offset / fps,
              "training_seed": seed, "training_seed_selection": "fixed by visualization_seed before rendering; not selected by test performance",
              "metric_aggregation_note": "Videos show one training seed; platform aggregate metrics separately average all training seeds with equal episode weight.",
              "video_horizon": horizon, "available_chart_and_csv_horizons": sorted(aligned),
              "frame_alignment": "episode frame 0 = initial state; frame k+1 = next_states for action step k",
              "forecast_alignment": "h-step forecast at target frame k starts from real state k-h and uses recorded actions k-h ... k-1; first h frames have no forecast",
              "plot_coordinates": "World XYZ coordinates in mm; Euclidean 3D error in mm; complete retrospective curves plus current video cursor",
              "controller": "fixed official motion planner with recorded joint-target perturbations; WM never controls the robot",
              "episode_selection": "all held-out test episodes, ascending episode id, without best-case selection",
              "episodes": mappings, "limitations": ["New recording run, not reconstruction of a past unrecorded run.",
                   "Privileged state model; video is simulator rendering, not a video predicted by the WM.",
                   "No online planning, update comparison, probability calibration or safety-effect claim."]}
    report_path = output / "visualization.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    paths.append(report_path)
    _write_html(output, report, paths)
    paths.append(output / "visualization.html")
    data.close()
    predictions.close()
    media = {".mp4": "video/mp4", ".png": "image/png", ".pdf": "application/pdf", ".csv": "text/csv",
             ".json": "application/json", ".html": "text/html"}
    preferred = {"simulation.mp4": 0, "comparison.mp4": 1, "visualization.html": 2,
                 "comparison_preview.png": 3, "visualization.json": 4}
    paths.sort(key=lambda path: (preferred.get(path.name, 5), str(path)))
    return [{"path": path.relative_to(output).as_posix(),
             "kind": "video" if path.suffix == ".mp4" else "image" if path.suffix == ".png" else "evidence",
             "media_type": media[path.suffix]} for path in paths]


def _write_html(output, report, paths):
    """An offline report using relative artifact paths; no external JS/CDN."""
    import html
    h, seed = report["video_horizon"], report["training_seed"]
    chart_links = []
    for row in report["episodes"]:
        episode = row["episode"]
        options = []
        for horizon in report["available_chart_and_csv_horizons"]:
            prefix = f"charts/episode_{episode:03d}_h{horizon}"
            options.append(f'<a href="{prefix}.png">{horizon} 步 PNG</a> · <a href="{prefix}.pdf">PDF</a>')
        chart_links.append(f'<tr><td>{episode:03d}</td><td><a href="{html.escape(row["source_video"])}">原始仿真录像</a></td><td>{" &nbsp; | &nbsp; ".join(options)}</td></tr>')
    downloads = " · ".join(f'<a href="predictions_h{horizon}_seed{seed}.csv">{horizon} 步完整逐帧 CSV</a>'
                           for horizon in report["available_chart_and_csv_horizons"])
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WM—ManiSkill 后果预测对照</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f7fa;color:#183047;font:16px/1.7 system-ui,sans-serif}}main{{max-width:1200px;margin:auto;padding:36px 24px}}
h1{{font-size:30px;margin:0 0 10px}}h2{{font-size:21px}}p{{color:#526172}}section{{background:white;border:1px solid #dde5ed;border-radius:12px;padding:24px;margin:24px 0}}
video{{width:100%;background:#192630;border-radius:8px}}a{{color:#0072b2}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #dde5ed}}.tag{{display:inline-block;background:#e8f2fa;padding:4px 12px;border-radius:6px;margin:4px 8px 4px 0;font-size:14px}}
</style><main><h1>小型状态 WM × ManiSkill</h1><p>真实仿真录像与预测后果同步对照。三种预测方法使用相同测试轨迹；机械臂由固定规划器执行。</p>
<span class="tag">Panda · PushCube-v1</span><span class="tag">全部 {len(report['episodes'])} 个留出测试回合</span><span class="tag">可视化训练种子 {seed}</span><span class="tag">{report['fps']:g} FPS · 1× 播放</span>
<section><h2>1. 预测对照画面</h2><video controls preload="metadata" poster="comparison_preview.png"><source src="comparison.mp4" type="video/mp4"></video>
<p>黑色：真实状态；蓝色：动作条件 WM；橙色：无动作 WM；绿色：状态保持基线。右侧展示世界坐标系下方块与末端的 XYZ 位置，及三维欧氏误差，单位均为 mm。</p>
<p>当前画面采用 {h} 步预测：目标帧 k 的预测从真实状态 k−{h} 出发，递归使用已记录的动作。竖线与仿真帧同步，曲线展示完整测试回合。前 {h} 帧没有对应预测，保留为空。该视频展示种子 {seed}；平台正式汇总指标另对全部训练种子取平均。</p></section>
<section><h2>2. 仿真录像</h2><video controls preload="metadata"><source src="simulation.mp4" type="video/mp4"></video>
<p>本次重新运行采集的测试回合，按回合编号依次播放。固定官方运动规划器执行，加同配置的关节目标扰动；WM 不参与控制。</p></section>
<section><h2>3. 可复核数据与逐回合图</h2><p>{downloads} · <a href="visualization.json">逐帧对齐说明</a></p><table><tr><th>回合</th><th>视频</th><th>一/多步预测图</th></tr>{''.join(chart_links)}</table></section>
<p>本实验验证状态后果预测链路。仿真录像不是世界模型生成的视频；结果不证明概率校准、防御效果、在线规划或更新效果。</p></main></html>'''
    (output / "visualization.html").write_text(document, encoding="utf-8")
