# TD-MPC2 × ManiSkill PushCube 正式验收报告

发布说明：本文的 `K:/world-model` 与 `/srv/world-model` 均为目录示例，需按机器调整。`results/`、`data-linux/`、`source-audit/` 和历史部署文档属于本地记录，未随仓库发布；文中的运行 ID、验收数值描述当时实测，不能据此假定新部署已有这些工件。

日期：2026-10-12。**工程验收通过。** 正式平台已完成真实训练、checkpoint 回载、三组闭环评测、固定动作潜空间预测、六段仿真录像、通用 predict/rollout 以及独立指标。范围是一个训练 seed、每组 10 个留出环境种子的受限预算基线，不是论文全量复现、广泛泛化或安全性结论。

机器可读总记录为 `results/tdmpc2-integration-20261012/acceptance/acceptance.json`（本地记录，未发布），相邻目录保存各 source/metric run 的 API 记录和日志。接入方法见 [中文开发教程](TD_MPC2_PUSHCUBE_GUIDE.zh-CN.md)，安装与排障过程见 [公开过程记录](TD_MPC2_PROCESS_LOG.zh-CN.md)。

## 1. 实际设置和算法来源

采用 ManiSkill 官方仓库 `examples/baselines/tdmpc2`，提交 `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`。复用官方 `TDMPC2`、WorldModel、Buffer、网络、损失、优化器和 MPPI；平台提供配置、固定回合环境桥、训练编排、工件及指标入口，上游 checkout 保持干净。

- 环境为 PushCube-v1、Panda、`physx_cuda`，输入为完整特权状态观测；本次实际维度 35，动作维度 8。
- 控制为 `pd_joint_delta_pos`，动作在 `[-1,1]` 内归一化；奖励为 `normalized_dense`。
- 每回合固定 50 个控制步，不因任务成功提前结束，分别报告“曾成功”和“最终成功”。
- 训练 seed 为 0，32 个并行环境，batch 256，每 8 条环境转移一次更新，官方默认 MPPI 参数。
- 三组评测共享留出 seeds 1000—1009。动作扰动组用同一最终权重，加标准差 0.1 的归一化动作高斯噪声，再裁剪到控制器范围。
- 三组观测噪声均为 0；每组录像固定选前两个种子 1000、1001，没有根据结果挑选视频。

本机为 RTX 4090 环境；原有其他 GPU 进程被保留。基础环境版本为 Python 3.11、Torch 2.7.0+cu128、ManiSkill 3.0.1、SAPIEN 3.0.3、Gymnasium 1.4.0。TD-MPC2 环境通过 `.pth` 引用基础环境，再安装补充依赖；它是依赖 base 的 overlay，并非可独立搬移的完整环境。

## 2. 模型任务与实际训练

以下七条 source/interface run 均已 `completed`：

| 内容 | 平台 run ID |
|---|---|
| 正式训练 | `run_d5fc89b69fb84542` |
| 初始检查点评测 | `run_9eefa45505254a41` |
| 训练后检查点评测 | `run_c1bbe3ad8f214a8e` |
| 训练后动作扰动评测 | `run_9ea6ef48e91f4787` |
| 固定动作潜空间预测 | `run_032261dca3894c71` |
| 通用 `task=predict` | `run_da83e6f77b034145` |
| 通用 `task=rollout` | `run_f3b7e259b9e645af` |

训练完成 **102,400 次真实交互、12,800 次更新、2,048 个回合**。模型参数量为 4,972,654；训练循环记录用时 **691.059 秒**。该计时来自训练报告，不含完整服务排队及所有初始化时间，不作为其他配置的速度保证。

最终 loss 均有限：total loss 0.074304417、consistency loss 0.000415388、reward loss 0.338190407、value loss 0.321776271。相对 `initial.pt` 的模型状态参数 L2 变化为 96.317518，确认不是只生成一个未更新的权重文件。

保存 `initial.pt`、四个中间 checkpoint 和最终 `model.pt`，并保留 `training_history.jsonl`、`training.json`、`config.resolved.json`、`provenance.json`。本次从随机初始化开始，`warm_start_checkpoint=null`。当前 checkpoint 包含模型、配置和元数据，不含优化器、replay 或 RNG；以后使用 warm start 也不是精确断点恢复。

前期 `gpu-smoke-r4` 的 1,600 次交互/200 次更新和单回合 `checkpoint-smoke` 仅用于排障与工程验证，未计入上述正式训练和评测。

## 3. 三组闭环结果

每组均为 10 个回合、每回合 50 步。return 是任务奖励之和，再对回合等权平均。

| 检查点/条件 | 平均 return | 曾成功 | 最终成功 | 平均步数 | 实际改变动作的步数 |
|---|---:|---:|---:|---:|---:|
| 本次初始权重，无干预 | 0.747131 | 0/10 | 0/10 | 50 | 0/500 |
| 训练后权重，无干预 | 40.453225 | 10/10 | 10/10 | 50 | 0/500 |
| 训练后权重，动作噪声 std=0.1 | 38.068812 | 10/10 | 10/10 | 50 | 500/500 |

扰动组的全部 500 个实际控制步均改变了动作，提议动作和执行动作均有保存；因此不是设置了参数却没有施加扰动。改变动作的步数不等同于发生裁剪的步数。结果说明在这一训练 seed 和这些留出场景上模型完成了任务，动作扰动组的成功率仍为 10/10、平均 return 较低；样本量和实验范围不足以支持广泛鲁棒性或安全提升结论。

本次没有测量碰撞、接触力或风险成本，任务成功率不能解释为安全率。观测噪声接口已存在，但没有进行正式观测噪声实验；可选逐分量扰动作用于混合单位 privileged state，不是经过校准的传感器噪声模型。

## 4. 固定动作潜空间预测

使用训练后 checkpoint 和无干预评测的真实轨迹，比较一步与五步窗口。模型从窗口初始观测编码出的 latent 出发，沿记录的实际动作递归预测；未来真实观测只在末端编码为评价 target，不反馈进入预测链。

| 指标 | H=1 | H=5 |
|---|---:|---:|
| 动作条件 latent MSE | 0.000240815 | 0.001306656 |
| latent persistence MSE | 0.002292892 | 0.006278533 |
| 同模型零动作 latent MSE | 0.002252745 | 0.006119405 |
| 窗口末步 reward MAE | 0.008759889 | 0.015236771 |

完整精度见 `acceptance.json` 和预测 run 的 `prediction_summary.json`，逐窗口数组保存在 `latents_episode_XXX_hH.npz`。reward MAE 对应窗口最后一步的奖励，不是 H 步累计回报误差。

这些是同一 checkpoint 表示空间内的对照。latent 没有物理坐标单位，不能解释为毫米误差、碰撞概率，也不能直接以不同 checkpoint 的 latent MSE 给模型排名。零动作使用同一模型作反事实动作对照，不是重新训练的无动作 WM；其 target 仍是实际执行动作的轨迹，也不是零动作真正执行后得到的仿真真值。当前没有物理状态解码头。

## 5. 视频、网页和通用接口

三组各两段 MP4，共 **6 段**，均已实际解码；每段 **51 帧、512×512**，包括 reset 后初始画面和 50 次控制之后的画面。保存真实轨迹、视频及 `visualization.html`，来源属于同一次仿真。

浏览器另检查了 **2 段**视频：均为 `readyState=4`、`error=null`、分辨率 512×512、时长 2.55 秒。HTTP 检查得到 HTML 200、视频 Range 请求 206。六段文件解码与两段浏览器加载是不同范围的检查，不能把浏览器结果扩写为六段逐一检查。

已有本机 SSH 隧道可访问 [训练后评测展示页](http://127.0.0.1:18765/api/v1/runs/run_c1bbe3ad8f214a8e/files/visualization.html)。它包含真实仿真视频和任务结果；链接依赖该隧道与服务器服务在线。

通用 `ModelAdapter.predict` 已实际返回 latent/reward 预测，通用 `ModelAdapter.act + EnvironmentAdapter` 已完成单回合 50 步 rollout，后者保存 `trajectory.json`。这证明通用接口可运行；由于原生与通用组件的生命周期和随机数调用顺序不同，不保证相同 seed 的逐步轨迹完全一致，也未将该单回合当作第四组 10 种子评测。

## 6. 独立指标与来源不变性

五条自动 `task=evaluate` 均为 completed，`metric_errors={}`，版本均为 `tdmpc2/1`：

| 来源 | 自动指标 run ID |
|---|---|
| 训练 | `run_73c77b116ccd426e` |
| 初始权重评测 | `run_06b1fa76398d4856` |
| 训练后评测 | `run_fc520f3ec5814287` |
| 动作扰动评测 | `run_a53d8023c19846d8` |
| 潜空间预测 | `run_1e3d87939d074219` |

指标在独立工作进程中读取来源输出，写到新评价目录。训练与控制指标从相应 JSON 汇总；预测指标在这一步读取并校验原预测 summary，没有再次加载模型作推理，不能将指标导入称为第二次独立神经预测复算。

另执行重复评价 **`run_27add12648d2402b`**，来源为训练后评测 `run_c1bbe3ad8f214a8e`。重复结果与先前自动指标完全相同，评价前后来源 output 的全部文件 SHA-256 不变。核验窗口从重复评价开始前计：此前为了响应式展示更新过 HTML，原 HTML 已单独备份；不能把这项只读评价检查描述成展示文件从模型任务结束后从未改变。

重复评价证据由本轮 `finalize_evidence.py` 保存为 `repeat-evaluation.json`。原自动评价保留，重复评价没有覆盖来源数据或原指标记录。

## 7. 测试和保存范围

- 本地完整套件：**49 passed、2 skipped**。两个跳过分别来自轻量平台环境缺少 NumPy 的旧可视化测试和可选 Gymnasium 模板测试；见 `results/tdmpc2-integration-20261012/pytest-full.xml`（本地记录，未发布）。
- 新增接线与指标验证：**2 项接线测试 + 16 项指标测试，共 18 passed**；见 `results/tdmpc2-integration-20261012/pytest-wiring.xml`（本地记录，未发布）。
- 真实 GPU 训练、权重回载、仿真、录像和 HTTP/浏览器证据单独记录，不以单元测试代替。
- 部署保留已有 43 条历史运行，未修改全局 CUDA 驱动或上游 checkout；只在项目缓存准备 PhysX GPU 库及私有驱动链接，并显式选择本机正确 Vulkan ICD。

正式服务器工件位于：

```text
/srv/world-model/deployments/wm-safetylab/data-linux/runs/<run_id>/output/
```

本地验收摘要、API 记录和日志位于：

```text
K:/world-model/deployments/wm-safetylab/results/tdmpc2-integration-20261012/acceptance/
```

本地目录是验收材料，不应默认称为服务器全部 checkpoint、视频和大型数组的完整镜像。具体文件以目录清单为准。

本轮另外保存了 `results/tdmpc2-integration-20261012/server-evidence.tgz` 及其解压目录：包含过程日志、依赖记录、原始轨迹、预测数组与视频，排除了 `.pt` 权重；六个训练 checkpoint 保留在服务器训练 run。可直接运行的 `examples/tdmpc2/use_existing_baseline.py` 还从 Windows Python 客户端实测完成预测，补充运行 ID 为 `run_8da23316e2ac4957`，记录见本轮 `python-client-predict.json`。

## 8. 已完成范围和剩余研究边界

已完成的是可复用的平台接入，以及记录配置下的单训练 seed、10 留出种子/条件工程基线。训练后模型在本组测试上完成任务，固定动作 latent/reward 对照和实际动作干预均产生可读证据。

没有实现或没有验证的内容包括：风险成本建模、校准安全概率、物理坐标解码、正式观测噪声研究、多训练种子不确定性、完整论文预算、跨任务/跨机器泛化，以及优化器/replay/RNG 精确恢复。环境 overlay 依赖既有 base，不能只复制新 venv 就宣称可移植。后续扩展应建立各自的数据、指标和对照，不把本轮任务成功直接命名为安全提升。
