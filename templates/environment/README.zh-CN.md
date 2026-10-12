# 接入另一个仿真环境：完整开发教程

本模板适配真实的 **Gymnasium Pendulum-v1**，演示如何在现有 WM-SafetyLab 0.3.0 中接入一个新仿真器：配置加载 → 独立 Python 工作进程 → 环境交互 → 轨迹 → 独立评价 → 可选录像。无需修改 `src/wms/core`、服务、网页或生产服务器注册表。

它是 CPU 单摆环境，不是机械臂；`ConstantPolicy` 只是恒定力矩的教学执行器，不是世界模型。要接论文模型或机器人任务，应替换适配器中的对应实现，而不是把这个示例当作安全研究 baseline。

## 1. 文件与职责

```text
templates/environment/
├── lab_env/
│   ├── adapter.py       # PendulumAdapter：创建、重置、步进、关闭，少量可选录像能力
│   ├── policy.py        # ConstantPolicy：为首次跑通环境提供一个明确的动作来源
│   ├── native.py        # 原生录像入口：保存轨迹，并显式登记 MP4
│   └── metrics.py       # 只读轨迹的评价函数
├── registry.json        # 组件名、Python 入口和两条实验配方
├── runtime.*.example.json # 每台机器自己的解释器、环境变量和超时
├── requirements*.txt    # 依赖范围与本机已验证版本
└── smoke.py             # 直接运行或向已有服务提交完整配置
```

四层职责对应：数据层记录初始观测、动作、真实转移；模型层由外部模型或教学策略提供动作；算法层决定使用 `rollout`、`predict`、`plan` 或 `native`；执行与反馈层在仿真器内执行，保存轨迹并交给独立评价任务。当前示例主要演示 S1/S5/S6，不会生成伪造的内部 WM 状态、规划结果或安全事件。

## 2. 最小环境接口

只需要普通 Python 类，无需继承平台基类：

```python
class MyEnvironment:
    def __init__(self, config: dict):
        # 在工作进程中导入仿真包、加载场景。
        ...

    def reset(self, seed: int, inputs: dict):
        # 只返回模型需要的初始观测；固定随机种子。
        return observation

    def step(self, action):
        return {
            "obs": next_observation,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            # cost/events/info 是可选字段，确实测得才填写。
        }

    def close(self):
        # 释放仿真器、渲染器或网络连接；允许重复调用。
        ...
```

Gymnasium 原始 `reset()` 返回 `(observation, info)`，`step()` 返回五元组。模板把它们转换成上述格式，并把 NumPy 数组转换成列表、标量转换成原生 `float/bool`。平台普通 `rollout` 会在 `finally` 中调用 `close()`，即使策略或环境报错也会释放资源。[官方 Env API](https://gymnasium.farama.org/api/env/)

**本例约定：**

| 字段 | 含义 |
|---|---|
| observation `[cosθ, sinθ, θdot]` | 前两项无量纲，第三项 rad/s；长度 3 |
| action `[torque_Nm]` | 力矩，单位 N·m；长度 1，范围 [-2, 2] |
| 物理时间步 | 0.05 s；录像采用 20 FPS 与物理时间对齐 |
| 重置 | `seed=42`；可在 `inputs.reset_options` 传官方支持的初始化范围 |
| 终止 | `terminated` 为任务终止；`truncated` 为环境时间限制；外层步数预算另记为 `step_limit` |
| reward | 原任务奖励，不等同于安全 cost；没有定义安全信号 |

数值和单位以 [Pendulum 官方说明](https://gymnasium.farama.org/environments/classic_control/pendulum/) 为准。适配器拒绝非有限值、错误形状或超范围动作，不悄悄修改模型的输出。

## 3. 安装到独立环境

本机已经验证 `K:/world-model/envs/wms-onboarding`，无需重复创建。以后接新仿真器时使用自己的环境；不要往正在使用的 ManiSkill、Wan 或 OpenVLA 环境中试装不兼容依赖。

Windows PowerShell，从项目根目录执行（首次创建新目录时）：

```powershell
. K:\world-model\use-k-drive.ps1
Set-Location K:\world-model\deployments\wm-safetylab
# 下面是新建环境示例；已有 wms-onboarding 时跳过创建。
& .venv\Scripts\python.exe -m venv K:\world-model\envs\my-simulator
& K:\world-model\envs\my-simulator\Scripts\python.exe -m pip install -r templates/environment/requirements.lock.txt
& K:\world-model\envs\my-simulator\Scripts\python.exe -m pip install --no-deps -e .
```

`--no-deps` 在这里仅安装工作进程可导入的平台代码。这个轻量环境可以跑本示例的工作进程；发 HTTP 请求请使用项目 `.venv`（已有 httpx），不要误以为所有 Web 依赖都已装进了仿真环境。

Linux 在你自己的服务器项目目录中执行：

```bash
cd /YOUR_PROJECT_ROOT/deployments/wm-safetylab
export TMPDIR=/YOUR_PROJECT_ROOT/tmp
export PIP_CACHE_DIR=/YOUR_PROJECT_ROOT/cache/pip
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
python3.11 -m venv /YOUR_PROJECT_ROOT/envs/my-simulator
/YOUR_PROJECT_ROOT/envs/my-simulator/bin/python -m pip install -r templates/environment/requirements.lock.txt
/YOUR_PROJECT_ROOT/envs/my-simulator/bin/python -m pip install --no-deps -e .
```

将 `/YOUR_PROJECT_ROOT` 全部替换成服务器已有的项目存储根目录。版本锁是 Windows Python 3.11 的实测组合，Linux 命令尚需在目标机器验证。无录像时只需要 Gymnasium/NumPy；录像额外需要 pygame-ce 和 imageio-ffmpeg，已包含在锁文件中。无需 CUDA/GPU。

## 4. 先单独跑通，再接网页

在本机现有环境验证无录像的标准路径：

```powershell
. K:\world-model\use-k-drive.ps1
Set-Location K:\world-model\deployments\wm-safetylab
& K:\world-model\envs\wms-onboarding\Scripts\python.exe templates/environment/smoke.py --output K:/world-model/deployments/wm-safetylab/results/my-pendulum-first
```

`smoke.py` 使用当前解释器执行工作进程。也可从平台环境调用，并通过机器配置选择仿真环境：

```powershell
. K:\world-model\use-k-drive.ps1
Set-Location K:\world-model\deployments\wm-safetylab
& .venv\Scripts\python.exe templates/environment/smoke.py --runtime templates/environment/runtime.windows.example.json --output K:/world-model/deployments/wm-safetylab/results/my-pendulum-runtime
```

每次换一个 `--output` 目录，脚本会拒绝覆盖已有配置。预期产物：

```text
results/my-pendulum-first/
├── config.json
├── source/trajectory.json
├── source/result.json
├── evaluation-config.json
└── evaluation/
    ├── metrics.json
    └── result.json
```

无 Web 的直接运行也会另开一个评价工作进程，源实验不会被评价覆盖。40 步默认配置的实测 return 为 `-237.23304250977984`；不要求跨所有依赖版本逐位相等。

## 5. 录像与工件登记

```powershell
. K:\world-model\use-k-drive.ps1
Set-Location K:\world-model\deployments\wm-safetylab
& .venv\Scripts\python.exe templates/environment/smoke.py --runtime templates/environment/runtime.windows.example.json --video --output K:/world-model/deployments/wm-safetylab/results/my-pendulum-video
```

机器配置中的 `SDL_VIDEODRIVER=dummy` 允许无窗口渲染。`--video` 选择 `template-pendulum-video` 原生配方，额外生成 `simulation.mp4` 和 `environment.json`。默认 40 步输出 41 帧：第 0 帧是初始状态，第 t+1 帧是第 t 次动作后的状态。视频时长约 2.05 秒，其中一帧展示初始状态。

为什么不用普通 `rollout` 自动录视频：当前核心只登记 `trajectory.json`，不会自动把仿真器内部写出的任意视频加入结果清单。因此这里用短小的 `native.py` 自己记录并返回 `outputs`。它在循环中调用 `context.check_cancelled()`，并在 `finally` 中关闭编码器和仿真器。你自己的原生入口也需要保留这两点。

## 6. 提交给平台，不要求重启现有服务

在**服务所在机器**解析注册表并提交完整配置；前提是该服务能访问配置中的解释器、模块和输出路径：

```powershell
. K:\world-model\use-k-drive.ps1
Set-Location K:\world-model\deployments\wm-safetylab
& .venv\Scripts\python.exe templates/environment/smoke.py --runtime templates/environment/runtime.windows.example.json --api http://127.0.0.1:19021 --video --output K:/world-model/deployments/wm-safetylab/results/my-pendulum-api
```

`19021` 是教程验证服务示例，请改为实际服务地址。该脚本先 `Registry(paths=[...]).resolve(...)`，再使用 `wms.run(config)` 提交已展开的入口、环境和运行配置，因此服务无需预先认识 `template-pendulum-video` 这个名字。完成后在运行列表查看本次任务；自动评价是另一个排队任务，需要等待它自己的状态。

**不能**在 Windows 上展开为 `K:/...` 的配置直接发给 Linux 服务器；API 不会自动上传代码、转换路径或创建虚拟环境。先把模板部署到 Linux，在该机器设置运行文件并展开配置，再发给服务器 API。这个限制同时适用于 SSH 隧道中的 `18765` 地址。

如希望永久出现在配方下拉列表，部署时给服务增加 `--registry /绝对路径/templates/environment/registry.json`，或将已经验证的组件配方放入自己的 `native/<名字>/recipes.json`。这属于部署步骤；本模板没有改动正在运行的生产服务。

## 7. 换成你要的仿真器：开发清单

| 要修改的位置 | 具体工作 | 首次交付应包含 |
|---|---|---|
| `adapter.py.__init__` | 替换仿真器创建、场景、机器人、控制模式；依赖在工作进程中导入 | 环境版本、任务 ID、资源路径、依赖锁 |
| `reset` | 固定种子，初始化任务，整理观测 | 初始观测示例与 reset 参数说明 |
| `step` | 转换动作，调用一次控制周期，转换返回值 | 观测与动作 shape/单位/坐标系/时间步 |
| `close` | 关闭渲染和仿真资源 | 正常结束、异常后均可关闭 |
| `metrics.py` | 根据真实轨迹读出任务/安全指标 | 指标定义、单位、缺失信号说明 |
| `native.py`（按需） | 多回合、批量或录像流程 | 工件清单、取消检查、文件写入当前运行目录 |
| `registry.json` | 改唯一组件 ID、入口和默认实验参数 | 至少一个 CPU/小规模 smoke 配方 |
| `runtime.<machine>.json` | 填该机器解释器和环境变量 | 能由指定解释器导入仿真器的证据 |

安全信号必须来自实际定义：例如碰撞事件要说明对象对、接触过滤和时间窗口；关节限位要说明上下限与容差。没有测量时像本例一样报告 `unavailable`，不要填 `cost=0` 或声称安全。奖励不是安全标签，单摆的转动也不能说明机械臂碰撞风险。

## 8. 如何接模型以及当前边界

- `rollout`：模型提供 `act(observation, instruction)`，环境提供 `reset/step/close`。
- `predict`：模型提供 `predict(request)`；配置 `inputs.actions`，同一初始状态下记录预测与仿真真实后果。输出在 `result.json` 的 `data.predicted`、`data.real` 和 `data.transitions`。
- `plan`：预测模型 + 规划器 + 环境 + 明确的候选动作范围与 cost 函数；仿真器不是规划器。
- 本例的三维单摆观测不能直接送给另一模板的一维位置模型；函数签名相同不代表观测语义、动作单位、时间尺度兼容。可用明确匹配 Pendulum 动力学的解析参考验证接口，之后换成学习得到的 WM。解析一致不构成学习能力证据。
- 当前一个普通配对任务由一个 `runtime.python` 执行，因此模型与环境需要能在同一环境导入。互相冲突的模型/仿真依赖需要额外进程通信，目前这个模板不实现模型服务。
- Gymnasium、旧 Gym、LIBERO 的返回接口不应混用：旧接口的 `done` 需要结合版本和时间限制信息转换；无法区分时要说明，不能凭空推断 `terminated/truncated`。
- ManiSkill GPU 批量观测是 Tensor，不能对多元素 Tensor 直接 `bool()`；需显式选择单环境、处理每个环境终止标志，并定义 batch 维度。自动 reset 会改变 terminal observation 的来源，评价需要真实最终状态而非下回合初始状态。本模板仅单环境，不宣称已解决这些批处理问题。
- 大图像/视频/点云不要全部嵌入 JSON；保存文件并返回平台现有 payload 或工件引用，记录轴顺序、单位和时间戳。恢复仿真状态用于候选反事实不是最小接口中的能力，需在具体仿真器上另行验证。

## 9. 已验证范围

2026-10-11，在 Windows Python 3.11、Gymnasium **1.4.0**、NumPy **2.4.6**、pygame-ce **2.5.8**、imageio-ffmpeg **0.6.0** 上实际运行：

- 标准 `rollout`：40 步，保存轨迹，独立评价无错误。
- 原生录像：同样的 40 步与奖励，保存 41 帧 MP4 和环境元数据，独立评价无错误。
- 证据目录：`results/onboarding-20261011/env-state/` 与 `results/onboarding-20261011/env-direct/`。

这验证了一个不同于 ManiSkill 的真实仿真器可以通过现有接口运行。它不是新机器人任务、学习型 WM 性能或安全防护效果的验证。
