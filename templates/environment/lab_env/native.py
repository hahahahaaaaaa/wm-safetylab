"""Optional recorder: native entrypoints explicitly register produced videos."""
from .adapter import PendulumAdapter
from .policy import ConstantPolicy


def record_episode(config: dict, output_dir, context):
    # A short native wrapper is appropriate for artifacts beyond core rollout.
    import imageio_ffmpeg

    context.check_cancelled()
    env_config = {**config.get("environment", {}).get("config", {}), "record_video": True}
    env = PendulumAdapter(env_config)
    writer = None
    try:
        policy = ConstantPolicy(config.get("model", {}).get("config", {}))
        obs = env.reset(int(config.get("seed", 0)), config.get("inputs", {}))
        metadata = env.metadata()
        dt = metadata["dt_s"]
        initial_frame = env.render_frame()
        height, width = initial_frame.shape[:2]
        writer = imageio_ffmpeg.write_frames(
            str(context.output / "simulation.mp4"), (width, height), fps=1.0 / dt,
            codec="libx264", pix_fmt_out="yuv420p", macro_block_size=1,
        )
        writer.send(None)
        writer.send(initial_frame)
        rows, end_reason = [], "step_limit"
        for step in range(int(config.get("steps", 40))):
            context.check_cancelled()
            action = policy.act(obs)
            transition = env.step(action)
            rows.append({"t": step, "time_s": step * dt, **transition,
                         "obs": obs, "action": action, "next_obs": transition["obs"]})
            obs = transition["obs"]
            writer.send(env.render_frame())
            context.event("progress", {"phase": "step", "step": step + 1})
            if transition["terminated"] or transition["truncated"]:
                end_reason = "terminated" if transition["terminated"] else "truncated"
                break
        context.check_cancelled()
        summary = {"steps": len(rows), "end_reason": end_reason,
                   "return": sum(row["reward"] for row in rows)}
        context.write_json("trajectory.json", {"steps": rows, "final_obs": obs,
                           "seed": int(config.get("seed", 0)), "summary": summary})
        context.write_json("environment.json", {**metadata, "frames": len(rows) + 1,
                           "frame_alignment": "frame 0 is reset; frame t+1 is next_obs at step t"})
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            env.close()
    return {"outputs": [
        {"path": "trajectory.json", "kind": "trajectory", "media_type": "application/json"},
        {"path": "environment.json", "kind": "metadata", "media_type": "application/json"},
        {"path": "simulation.mp4", "kind": "video", "media_type": "video/mp4"},
    ], "data": summary}
