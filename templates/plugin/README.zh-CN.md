# S1–S6 插件模板

适用 WM-SafetyLab 回合内核（`wms.episodes.kernel`）与插件框架（`wms.plugins`），2026-10-18。

一个插件就是一个 Python 类，再加一条注册和一个 CPU 测试，不用改模型代码。它挂在闭环的某一个环节上：要么改这里流过的东西（故障、攻击、防御），要么只看不改（监控）。改完的东西和插件写下的事件都会记进每步的 `trace.jsonl`。

## 五步上手

1. 把 `templates/plugin/` 复制到自己的开发目录，把包名 `wms_plugin_template` 和注册名 `template-*` 都改成自己的前缀，避免与别人冲突。
2. 在 `gripper_flip.py` 的位置写自己的类：设好 `stage`、`role`，只实现本环节的钩子。
3. 在 `registry.json` 的 `plugins` 里登记：`entrypoint`、`stage`、`role`、默认 `config`；S2–S4 插件还要写 `side` 和 `requires`，攻击插件要写 `threat`。服务提交时就按这条登记做检查。
4. 跑一致性测试（几秒）：

   ```powershell
   . K:\world-model\use-k-drive.ps1
   Set-Location K:\world-model\deployments\wm-safetylab
   .\.venv\Scripts\python.exe -m wms plugin test <你的注册名> --registry <你的目录>\registry.json
   ```

   `--config` 可以给一段 JSON，或一个 JSON 文件的路径。测试用 `native/toy/runtime.json` 里那个带 numpy 的 Python 跑玩具环境，结果写在 `test-output/plugin-test/<注册名>/report.json`。
5. 在配方里挂上插件，到任务卡指定的真实模型上跑一次：

   ```json
   "params": {"plugins": [{"plugin": "<你的注册名>", "config": {"p": 0.2}}]}
   ```

## 环节与钩子

| 环节 | 钩子 | 收到什么 | 返回什么 |
|---|---|---|---|
| ENV | `on_env(t, ctx)` | 第 t 步开始；`ctx.env` 是环境本身 | 不返回；改了世界就 `ctx.log(...)`（内核据此重新渲染） |
| S1 | `on_observation(obs, ctx)` | 将送进模型的观测（`obs["rgb"][相机]` 为 HxWx3 uint8） | 新观测 |
| S1 | `on_instruction(text, ctx)` | 语言指令 | 新指令 |
| S2 | `on_state(state, ctx)` | 模型的记忆或潜状态；`ctx.phase` 为 `"read"`（读之前）或 `"written"`（写之后） | 新状态，或原样返回 |
| S3 | `on_prediction(pred, ctx)` | 一次预测；`ctx.meta["kind"]` 为 `"rollout"`（规划里的每步推演）或 `"chosen_next"`（选中动作的下一步预测，只作报告） | 新预测 |
| S4 | `on_plan(plan, ctx)` | `{"candidates", "scores", ...}` | 改 `scores` 即重排候选 |
| S5 | `on_action(action, ctx)` | 将送进仿真的动作 | 新动作；防御可调 `ctx.intervene("stop" 或 "fallback", action=...)` 接管 |
| S6 | `on_feedback` / `on_update` | 第二轮（P2）才接入，目前内核会拒绝 | |

所有插件都可以另外实现：`reset(episode, ctx)`（每回合开始）、`after_step(step, ctx)`（仿真步之后，只读，`step` 含建议动作、执行动作、奖励、事件）、`end_episode(ctx)`（返回的字典写进回合汇总 `plugin_reports`）、`close()`。

同一钩子上的执行顺序：先 fault / attack，再 defense，最后 monitor。所以防御看到的是被攻击后的值，监控看到的是最终值。

## ctx 能用什么

| 名称 | 说明 |
|---|---|
| `ctx.t`、`ctx.episode`、`ctx.seed` | 步号、回合序号、回合种子 |
| `ctx.rng` | 本插件自己的随机数（numpy Generator，按回合种子和插件位置设定）。不要用 `numpy.random` 或 `random` |
| `ctx.config` | 本插件的配置 |
| `ctx.spec` | 环境规格：动作维度与范围、相机、步数上限、`safety`（危险判据的阈值） |
| `ctx.log(dict 或 key=value)` | 写一条事件到本步 trace。S1 / ENV 的 fault、attack 写的事件也会进 `steps.jsonl` 的 `perturbations` / `target_moves` 列 |
| `ctx.truth()` | 仿真真值（`env.get_state()`），需要 `privileged = True` |
| `ctx.hazards` | 上一步之后成立的危险谓词，需要 `privileged = True` |
| `ctx.locate(name)` | 物体在图像中的框，需要 `privileged = True` 且 `requires` 含 `"image_box"` |
| `ctx.env` | 环境本身，只给 ENV 插件 |
| `ctx.intervene(kind, action=None, reason="")` | 只给 S5 防御：`stop` 本步动作不执行并结束回合；`fallback` 改执行给定动作 |

监控插件读 `ctx.truth()` 和 `ctx.hazards` 不需要声明 `privileged`。

## 必须遵守的规则（`wms plugin test` 会检查）

- 不要原地修改收到的值，返回一个新值。测试在严格模式下运行，原地修改会直接报错。
- 随机数只用 `ctx.rng`。测试会检查全局随机状态有没有被动过。
- 只实现自己环节的钩子。
- 防御默认不能读仿真真值。确实要读，写 `privileged = True` 并在配置里设 `"oracle": true`，结果会标成 oracle 上界。
- 攻击必须有威胁模型卡 `threat`：`knowledge`（`black_box` / `gray_box` / `white_box`）、`access`（从哪里进入）、`budget`（预算）、`goal`（目标），可选 `adaptive`（是否知道已挂的防御）。只有预算和知识相同的攻击才能比较成功率。
- 监控不能改变任何动作。测试会和不挂插件的运行逐步对比。
- 模块顶层只 import 轻量的东西（`wms.plugins`、标准库）。numpy、torch 放到钩子里 import，或在注册条目里写全 `stage` / `role` / `requires`，因为服务的 Python 里没有这些库。

## 各模型提供哪些探针

S2–S4 插件要用 `requires` 声明需要的探针。模型不提供，提交时就会被拒绝。

| 模型（注册名） | 探针 | 值的格式 |
|---|---|---|
| 玩具 `toy-centroid` / `toy-centroid-memory` | state | `{"cube": [x, y] 或 None, "goal": ..., "age": {...}}` |
| 玩具 `toy-planner` | state、predict、plan | state：`{"latent": (7,), "memory", "plan_mean"}`；predict：`(N, 7)` 或 `(7,)`；plan：`candidates (N, H, 2)`、`predicted (N, H, 7)`、`scores (N,)` |
| TD-MPC2 `tdmpc2-pushcube` | state、predict、plan | state：`{"latent": (512,), "plan_mean": (H, A) 或 None}`；predict：`(N, 512)`；plan：每轮 MPPI 一次，`candidates (N, H, A)`、`scores (N,)`。均为 float32 numpy |
| MemoryVLA `memoryvla-bridge` | state | `{"timestep", "cognitive": [(t, tensor)], "perceptual": [...]}`，GPU 上的 bf16 torch 张量，返回新列表，不要原地改张量 |
| OpenVLA `openvla-bridge` | 无 | 只能挂 ENV、S1、S5 |

玩具 `toy-planner` 的三个探针与 TD-MPC2 含义相同。S2–S4 插件可以先在它上面写测试，再上 GPU。

## 输出在哪里

- `episodes/ep_XXX/trace.jsonl`：每步一行，包括各环节跑了哪些插件、插件事件、建议动作与执行动作、接管、危险谓词。
- `episodes/ep_XXX/steps.jsonl`：原有格式不变。执行的动作在 `action` 列；被 stop 的那一步 `action` 为 `null`，事件为 `plugin_stop`。
- `episodes.json`：每回合汇总，另有 `takeovers`、`hazard_steps`、`plugin_reports`。
- `run_manifest.json`：本次挂的插件及其配置。

任务卡模板见 [TASK_CARD.zh-CN.md](TASK_CARD.zh-CN.md)。
