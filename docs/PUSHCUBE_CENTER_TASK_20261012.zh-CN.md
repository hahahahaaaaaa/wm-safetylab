# PushCube 圆心精度任务：实验计划与实现

日期：2026-10-12。任务 ID：`WMS-PushCubeCenter-v1`，奖励版本 `center-dense/1`。这是平台自定义精度控制任务，不是 ManiSkill 官方 PushCube-v1 的新成绩。

## 固定任务定义

- 保留原 Panda、35维完整状态、8维 `pd_joint_delta_pos` 动作、目标初始化和20cm水平推动距离。`goal_radius=.1` 保持不变，因为官方同时用它设置场景与目标初始位移。
- 中心任务成功：方块中心到目标中心的XY欧氏距离不超过0.02m，且 `abs(cube_z - .02) <= .005m`。
- 每回合50步、20Hz。分别报告曾成功、最终成功及末5个控制步持续成功；末5步指标排除reset状态，不意味着更长时间稳定或真实机械安全。
- 原官方10cm区域成功仅作参考单列，不能冒充新任务成功。

## 奖励

令 `d` 为圆心XY距离，`p` 为末端到方块后方推动点的距离，`T`为方块在桌面高度容差内：

`r = 0.2*(1-tanh(5*p)) + T*(0.3*(1-tanh(5*d)) + 0.5*(1-tanh(40*d)))`

这是normalized reward，取值在0到1之间。即使已进入10cm或2cm区域，也继续随圆心距离改善；没有“成功后直接置最大值”的覆盖逻辑。dense reward为其4倍。不是在原日志上重写reward，而是在新任务中真实交互和重新训练。

## 本轮训练与评测安排

- 从随机权重开始，训练seed=0，32并行环境，5M模型，batch256，每8条转移一次更新。
- 首轮预算204800真实环境转移，每51200步留checkpoint；不把1600步部署检查计入正式训练。
- 真实训练录像预先选定env0在开始、约25%、50%、75%、最后一批回合；带探索噪声，单独标为训练采集录像，不作为独立成功率。
- 用开发种子1100–1104查看检查点表现；测试种子2000–2019只用于最终冻结检查点评测，全部录制，保留失败。
- 首轮如果未学会中心控制，可明确记录追加预算/新配置；不能通过换种子或剪掉失败视频让结果看似通过。一次训练seed不支持广泛泛化结论。
- 正式评测报告实际最终误差、末5步稳定率、原区域成功率与中心成功率。任务成功不是安全率。

## 平台接入

- `src/wms/scenarios/maniskill/push_cube_center.py`：独立任务注册、判据与reward。
- `src/wms/engines/tdmpc2/compat.py`：选择任务、场景构建、任务/奖励标识和checkpoint绑定。
- `src/wms/engines/tdmpc2/backend.py`：训练及训练过程录像；没有改变上游网络、损失或MPPI。
- `src/wms/engines/tdmpc2/native.py`：评测、轨迹、中心判据和HTML展示。
- `src/wms/engines/tdmpc2/media.py`：原MP4、兼容WebM及末帧PNG。
- `src/wms/engines/tdmpc2/metrics.py`：独立结果汇总，旧报告缺测字段不补0。
- `native/tdmpc2_center/recipes.json`：独立训练、评测及CPU汇总报告入口，训练与评测复用已有TD-MPC2机器环境。

本轮部署记录保存在项目的 `results/pushcube-center-20261012/`。服务器的原生依赖、旧官方任务、历史权重和结果均保留。运行ID和验收数值将根据实际结果追加。

## 重复运行方法

无需另装模型依赖：复用已配置的 `native/tdmpc2/runtime.json` 和相同的GPU原生环境。服务需要在没有活动实验时重启一次，读取新增配方；之后修改每次实验参数不必重启。

在已连接服务的 Python / CLI 控制环境中，Windows 隧道使用 `WMS_BASE_URL=http://127.0.0.1:18765`，服务器本机使用 `http://127.0.0.1:8765`（不加 `/api/v1`）。PowerShell 可设置 `$env:WMS_BASE_URL="http://127.0.0.1:18765"`。然后：

```python
import wms
training = wms.run({"recipe": "tdmpc2-pushcube-center-train"})
training = wms.wait(training["id"])
assert training["state"] == "completed", training.get("error")
evaluation = wms.run({
    "recipe": "tdmpc2-pushcube-center-eval",
    "inputs": {"source_run": training["id"]},
})
print(evaluation["id"])
```

CLI 启动训练：`wms run tdmpc2-pushcube-center-train --wait`。评测保存如下JSON后使用 `wms run center-eval.json --wait`，将示例运行ID替换为你刚完成的训练：

```json
{
  "recipe": "tdmpc2-pushcube-center-eval",
  "inputs": {"source_run": "run_REPLACE_WITH_TRAIN_ID"},
  "params": {"episodes": 20, "seed_start": 2000, "video_episodes": 20}
}
```

若需要训练前对照，在同一 `source_run` 下增加 `inputs.initial_checkpoint=true`。旧官方任务权重会被中心专用评测入口拒绝，不会偷偷改成原10cm任务。模型和仿真器都在真实原生Python进程运行，视频来自真实渲染；没有手写轨迹代替模型决策。

网页中选择“圆心精度 2cm 独立任务训练”；完成后在中心评测配方中指定该训练运行或model.pt。查看输出的 `visualization.html`，或最终汇总运行的 `experiment.html`。每段录像同时提供WebM、MP4与末帧；如果内置播放器不支持某编码，可以切换来源或查看末帧。
## 实际完成结果（2026-10-12）

正式训练从随机权重开始，在 RTX 4090 上完成204800条真实环境转移、4096个训练回合和25600次梯度更新，耗时1383.61秒（约23.06分钟）。模型参数4972654；参数变化L2为130.80，损失全程有限。没有追加训练预算，没有用测试集选择检查点。

开发种子1100–1104的最终模型结果为5/5成功、5/5末5步稳定，平均误差4.86mm。开发评测前保存的选模规则为至少4/5最终中心成功；本次满足规则后冻结最终model.pt，再打开测试种子2000–2019。

| 同一中心任务、同20个测试种子 | 初始权重 | 训练后最终权重 |
| --- | ---: | ---: |
| 最终2cm中心成功 | 0/20 | 18/20（90%） |
| 最后5步持续中心成功 | 0/20 | 18/20（90%） |
| 全回合最终平均误差 | 200.03mm | 6.57mm |
| 最终最大误差 | 200.32mm | 37.98mm |
| 原10cm区域标准参考 | 0/20 | 20/20 |

训练后仍有两个失败回合：seed=2002，误差23.87mm；seed=2012，误差37.98mm。两者均保留在均值、成功率、逐回合表格与视频中。本结果说明中心任务的奖励与重训练有效，但并不代表100%可靠、数学意义零误差或机械臂安全已经获得验证。训练随机种子仅1个；更广初态、多个训练种子与扰动条件尚未开展。

### 运行和证据入口

| 阶段 | 运行ID | 入口 |
| --- | --- | --- |
| 正式训练 | `run_5865dede171646ad` | [训练记录](http://127.0.0.1:18765/api/v1/runs/run_5865dede171646ad/files/training.json) |
| 开发评测 | `run_ea6304f04d3647d7` | [5回合开发评测](http://127.0.0.1:18765/api/v1/runs/run_ea6304f04d3647d7/files/visualization.html) |
| 冻结权重测试 | `run_cb249f1199f542e6` | [20回合完整测试和视频](http://127.0.0.1:18765/api/v1/runs/run_cb249f1199f542e6/files/visualization.html) |
| 初始权重对照 | `run_f6a3b5d366e14655` | [20回合初始对照和视频](http://127.0.0.1:18765/api/v1/runs/run_f6a3b5d366e14655/files/visualization.html) |
| 汇总报告 | `run_6cea49065a5f49db` | [训练过程、曲线、前后对照及全部测试视频](http://127.0.0.1:18765/api/v1/runs/run_6cea49065a5f49db/files/experiment.html) |

本机18765端口通过隧道访问服务器；在服务器本机可把链接端口改为8765。GitHub仅保存实现与说明，实际权重、轨迹、视频仍保存在服务器的 `data-linux/runs/<run_id>/output/`。

正式研究记录包含50段真实回合录像：5段预定训练片段、5段开发评测、20段最终测试、20段初始权重对照。每段51帧（含reset画面）、20fps、2.55秒，提供MP4、WebM与末帧PNG。5段训练录像只覆盖预先选定的环境0回合，**不是23分钟训练期间的连续全量录像**。汇总页直接呈现5段训练与20段测试，另外两组通过链接查看；没有挑选最佳测试视频。

| 训练片段开始步 | 最终圆心误差 | 中心任务成功 |
| ---: | ---: | --- |
| 0 | 199.69mm | 否 |
| 51200 | 8.89mm | 是 |
| 102400 | 10.14mm | 是 |
| 153600 | 12.48mm | 是 |
| 203200 | 12.94mm | 是 |

训练片段使用带探索的策略，不能拿这些5个片段计算独立测试成功率。视频展示实际仿真渲染；模型输入是35维状态，并不是从视频直接控制。

### 复现与实现核对

- 最终权重SHA-256：`643b0e41479100d88005b46a2540a29e4f5b1bfe617fce0a56153a9811d0559b`。
- 中心任务组合源码SHA-256：`f6d9bf8386e1094258f2080753b8917fd4c47a5df162bb6b4bcf2f47bc267911`。
- 实际训练源码快照：服务器 `results/pushcube-center-20261012/training-source/`；来源清单、固定测试协议、各阶段配置和状态同目录留存。各训练输出还保存resolved config、provenance、initial.pt、阶段检查点、model.pt、训练历史和录像清单。
- 报告读取各来源文件，只在自己的运行目录写入 `experiment.html` 和 `evidence_summary.json`；不会改写原训练、测试数据。报告同时核对权重所属训练目录及实际SHA，防止把别次训练误标为本次初始或最终权重。
- 中心功能专项44项测试通过；当前完整平台套件141项通过、7项缺可选依赖而跳过。另在服务器真实检查初始化20cm位移、桌面高度约束、8cm不算中心成功、距离更小奖励更高，以及1600步训练/视频部署检查。
- 使用独立无头Edge检查汇总页：25个内嵌视频、无页面脚本错误；抽查起始训练与最后测试视频可播放、512×512、播放时间推进。内置应用浏览器的实际解码能力仍以用户设备为准，提供WebM/MP4双格式及末帧备用。

汇总报告也是独立配方，可以从Python重复生成：

```python
report = wms.run({
    "recipe": "tdmpc2-pushcube-center-report",
    "inputs": {
        "training_run": "run_5865dede171646ad",
        "test_run": "run_cb249f1199f542e6",
        "initial_run": "run_f6a3b5d366e14655",
        "development_runs": ["run_ea6304f04d3647d7"]
    }
})
print(wms.wait(report["id"]))
```
