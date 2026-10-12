# TD-MPC2 × ManiSkill PushCube 接入过程

日期：2026-10-12。本公开副本由当时的本地过程记录整理，移除了个人用户名和内网连接信息；实验数字与边界保持原样。最终结果见[正式验收报告](TD_MPC2_PUSHCUBE_ACCEPTANCE_20261012.zh-CN.md)。

发布说明：本文的 `K:/world-model` 与 `/srv/world-model` 均为目录示例，需按机器调整。`results/`、`data-linux/`、`source-audit/` 和历史部署文档属于本地记录，未随仓库发布；文中的运行 ID、验收数值描述当时实测，不能据此假定新部署已有这些工件。 原记录位于 `results/tdmpc2-integration-20261012/PROCESS_LOG.zh-CN.md`（本地记录，未发布）；下文未带根目录的证据文件均相对该目录，属于未发布的本地记录。

## 1. 接入前检查

- 本地平台：`K:\world-model\deployments\wm-safetylab`。
- 服务器：`<deployment-server>`，项目 `/srv/world-model/deployments/wm-safetylab`。
- 平台 API v0.3.0；已有 43 条运行记录，队列空闲。
- RTX 4090 24 GB，另一个用户任务占用约 3.5 GB，保留该进程。
- 已有 `maniskill-pushcube` 环境：Python 3.11、torch 2.7.0+cu128、ManiSkill 3.0.1、SAPIEN 3.0.3、Gymnasium 1.4.0。
- 原有 ManiSkill 教学链路使用 CPU 物理仿真和自定义状态。新基线单独采用 GPU 物理仿真、完整 35 维状态、`pd_joint_delta_pos` 8 维动作，不能直接复用原来的状态模型权重。

## 2. 选择和锁定源代码

- 使用 ManiSkill 官方仓库的 `examples/baselines/tdmpc2`，而非另造同名简化模型。
- 提交：`62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`。
- 服务器干净源目录：`/srv/world-model/repos/ManiSkill-tdmpc2-20261012`。
- 官方 `TDMPC2`、`WorldModel`、`Buffer`、网络、损失、优化器和 MPPI 复用；平台提供独立循环及 Gymnasium 兼容边界，不改上游 checkout。
- 源码核对见项目 `source-audit/AUDIT.md`（本地记录，未发布）。

## 3. Python 环境

- 新建 `envs/tdmpc2-maniskill-20261012`，通过 `.pth` 复用已验证 ManiSkill 环境的大型 CUDA/物理依赖。
- 新环境安装 hydra-core 1.3.2、hydra-submitit-launcher 1.2.0、termcolor 2.4.0、tensordict 0.8.3、torchrl 0.8.1、moviepy 1.0.3、wandb 0.19.11。
- `pip check` 通过；完整包记录 `requirements-effective.lock.txt`。
- 这是依赖已有环境的 overlay，不能声称直接打包该 venv 即可迁移。新机器应重建基础环境或依据完整依赖记录重装。

## 4. 实际遇到的部署问题

| 尝试 | 现象 | 处理与证据 |
|---|---|---|
| 首次 GPU probe | SAPIEN 自动下载 PhysX GPU 库卡住，默认位置为用户目录 | 停止本次 probe；库改存项目 `cache/physx`，兼容层显式加载 |
| 服务器 curl | 网络慢，短超时重试不能完成 74.3 MiB 下载 | Windows 下下载到 `K:\world-model\downloads`，再 SCP 到服务器项目目录 |
| 下载确认 | SHA256 `167a01aad7381afef963b89169968c289e7b653880a7a823c116d87ee5c00fc6` | ZIP 解压到项目 `cache/physx/105.1-physx-5.3.1.patch0` |
| `gpu-smoke-r2` | `PhysxGpuSystem: CUDA failed` | `diagnose_gpu.py` 测得 `libcuda.so` 的 `cuInit` 返回 34，系统优先加载 `/usr/local/lib/libcuda.so` 占位库 |
| 驱动修复 | PyTorch 能用 CUDA，PhysX 仍不能用，原因是库名解析路径不同 | 项目 `cache/nvidia-driver/libcuda.so` 指向真实驱动，通过运行环境 `LD_LIBRARY_PATH` 优先加载；修复后 cuInit=0、PhysxGpuSystem 创建成功 |
| `gpu-smoke-r3` | `ErrorIncompatibleDriver` | 诊断命令错误指定 `/usr/share/vulkan/icd.d/nvidia_icd.json`；本机实际正确位置是 `/etc/vulkan/icd.d/nvidia_icd.json`，机器配置已固定 |
| `gpu-smoke-r4` | 真实物理仿真、回放、梯度更新通过 | 1,600 次交互，200 次更新，32 episodes，4,972,654 参数；最终损失有限，参数 L2 变化 13.357929；不是正式表现评估 |
| `checkpoint-smoke` | 独立进程加载 probe 权重并录制真实仿真 | 一回合评测、NPZ 轨迹、MP4 和 HTML 输出成功；正式评测另走平台队列 |

没有修改全局驱动、原生 ManiSkill 环境或上游模型源代码。

## 5. 平台接入

- 新增 `src/wms/engines/tdmpc2/{compat,backend,native,adapter,metrics}.py`。
- 新增 `native/tdmpc2/recipes.json` 三条原生配方：训练、检查点评测、固定动作潜空间预测。
- 机器运行配置选择 TD-MPC2 环境，服务自身继续使用原 `.venv`。
- 队列空闲后，通过 `refresh_service.py` 按原命令重启服务，并更新服务身份记录。
- 核验历史 ID 集合完全一致，43 条历史记录保留，三条新配方均 `configured=true`。见 `service-refresh.json`。
- 运行使用的接入代码及配方快照保存到 `source-snapshot/`，摘要见 `source-snapshot.sha256`。

## 6. 正式实验协议与提交

固定训练：seed 0、5M 模型、32 个环境、102,400 transitions、batch 256、每 8 transitions 一次更新、50 步回合、官方默认 MPPI 参数。它是受限预算单种子研究基线，未声称复现官方完整 1M-step 成绩。

| 实验 | 平台运行 ID | 作用 |
|---|---|---|
| 正式训练 | `run_d5fc89b69fb84542` | 在线收集、回放、真实更新与检查点 |
| 初始检查点评测 | `run_9eefa45505254a41` | 相同结构与规划预算的未训练对照 |
| 训练后检查点评测 | `run_c1bbe3ad8f214a8e` | 留出场景中的任务表现 |
| 动作扰动评测 | `run_9ea6ef48e91f4787` | 同一训练后模型，归一化动作高斯噪声 std=0.1，执行前裁剪并计数 |

三组评测均使用 seeds 1000–1009、每回合 50 步，分别报告曾经成功与最终成功。噪声采用独立随机数生成器，保存 proposed/applied action。奖励是任务奖励；未测量碰撞、接触力等安全信号。

## 7. 本地检查

`pytest -q --basetemp results/tdmpc2-integration-20261012/pytest-full --junitxml results/tdmpc2-integration-20261012/pytest-full.xml`：49 passed，2 skipped，39.18 秒。

跳过原因：平台轻量 Windows 环境没有 numpy，旧可视化测试跳过；Gymnasium 是可选 worker 依赖，旧模板对应测试跳过。TD-MPC2 的 16 项独立指标测试通过。真实 GPU 运行证据独立记录，不能以单元测试替代。

新增的两项平台接线测试验证 run ID 定位初始/最终 checkpoint、评价轨迹，以及 GPU override 不修改机器文件；与 16 项指标测试一起执行，18 passed，见 `pytest-wiring.xml`。

## 8. 最终实跑和验收

正式训练 completed：102,400 transitions，12,800 gradient updates，2,048 episodes，691.059 秒，参数 L2 变化 96.3175，全部记录 loss 有限。上游源码仍干净。

| 组别 | 留出种子数 | 曾成功/最终成功 | 平均回报 |
|---|---:|---:|---:|
| 初始模型 | 10 | 0/10；0/10 | 0.747131 |
| 训练后模型 | 10 | 10/10；10/10 | 40.453225 |
| 训练后模型 + 动作噪声 0.1 | 10 | 10/10；10/10 | 38.068812 |

动作扰动 500/500 步实际改变 applied action。三组共 30 条真实轨迹，全部包含 51×35 状态和 50×8 动作，值有限；保存 6 段 MP4，每段 512×512、51 帧、20 FPS、2.55 秒。

固定动作预测 `run_032261dca3894c71` completed。h=1/h=5 共 20 组 NPZ；根据保存数组使用 NumPy 独立重算，与 Torch 阶段报告一致。h=1 动作条件 latent MSE 0.000240815，保持基线 0.002292892，零动作同模型 0.002252745；h=5 分别 0.001306656、0.006278533、0.006119405。仅是同一 encoder/checkpoint 内的潜空间误差。

通用接口：`run_da83e6f77b034145`（predict：3 个候选动作同时取得潜空间预测与真实仿真观测），`run_f3b7e259b9e645af`（rollout：50 步官方 MPPI 闭环）均 completed。它们验证普通模型/环境组件路径，正式成绩仍以上述 native 评测协议为准。

五项自动指标任务均 completed：训练 `run_73c77b116ccd426e`；初始 `run_06b1fa76398d4856`；训练后 `run_fc520f3ec5814287`；扰动 `run_a53d8023c19846d8`；预测 `run_1e3d87939d074219`。

重复评分 `run_27add12648d2402b` completed，数值与训练后自动评分完全相同，来源目录全部文件 SHA256 不变。评分前仅重生成三个报告的响应式 HTML，使窄侧栏能完整显示录像；旧 HTML 单独保存到 `reports-before-responsive-layout/`。

浏览器实测：报告 DOM 正确，两个视频 readyState=4、error=null、512×512、duration=2.55。HTML GET 返回 200；MP4 Range GET 返回 206。该 API 未提供文件 HEAD 路由，最初 HEAD 404 不代表视频不存在，随后已改用真实 GET 验证。

源运行 ID、指标 ID、逐组数值、视频检查及通用接口检查见 `acceptance/acceptance.json`。重复评分摘要见服务器证据包内 `repeat-evaluation.json`。`server-evidence.tgz` 包含原生过程日志、依赖锁文件、来源、原始轨迹、预测数组和视频；**不包含 `.pt` 权重**，六个训练 checkpoint 保留服务器训练 run 的 output 目录。

验收后队列空闲，本次所有 GPU 工作进程已退出，原有其他用户进程保留；服务运行正常。服务身份文件已更新为新 PID，旧身份记录单独保存。

补充执行：Windows `.venv` 使用 `examples/tdmpc2/use_existing_baseline.py` 通过现有 SSH 隧道提交真实预测，`run_8da23316e2ac4957` completed。保存客户端返回为 `python-client-predict.json`；这是可运行教程脚本的实测，未计入三组正式评测统计。

## 9. 结论边界

在本机、记录版本和预算下，建立了真实可训练、可加载、可运行动作干预、可记录和重评分的 TD-MPC2—PushCube 基线。这里只训练 seed 0，并评测 10 个留出种子。未进行多训练种子统计、官方完整预算复现、校准概率预测、风险成本训练或碰撞/接触力安全验收；未验证精确断点续训和独立机器迁移。零动作对照不是重新训练的无动作模型。
