# 小型状态 WM＋ManiSkill：仿真录像与同步预测对照

发布说明：本文的 `K:/world-model` 与 `/srv/world-model` 均为目录示例，需按机器调整。`results/`、`data-linux/`、`source-audit/` 和历史部署文档属于本地记录，未随仓库发布；文中的运行 ID、验收数值描述当时实测，不能据此假定新部署已有这些工件。

本次是在原有工程实验上增加可选录像和定量展示。机械臂仍由固定官方规划器执行，小型状态 WM 离线预测动作后果，未增加 WM 闭环控制。

状态：2026-10-11 已完成真实服务器运行、独立评价、逐帧审计和浏览器加载检查。

## 实测验收

- 模型运行 `run_d57f3484d13949ec` 与自动评价 `run_d37d3bf23a1544a9` 均为 `completed`，0 个指标错误。
- 新采集 2217 条转移；六个数据数组与原实验逐元素完全一致；18 个独立评价指标也与原评价逐项一致。
- 6 段原始录像分别为 61、61、66、57、62、61 帧；两段合集均为 368 帧、20 FPS、18.4 秒，实际解码核验。
- 对 seed 0 的两份 checkpoint 独立重做一步与五步推理，与保存结果完全一致；两份 CSV 共 2208 行完成逐行复算。
- 49 个结果文件均已登记到平台；浏览器确认两段 MP4 已加载、时长与分辨率正确，没有媒体错误。
- 核心测试 31 项通过；新增 9 项数值对齐测试在已有 NumPy 环境中全部通过。轻量平台环境没有 NumPy，运行完整核心套件时该数值测试模块会跳过。
- 部署保留了 41 份旧结果快照、机器配置与原生环境；正式审计未修改新旧实验工件。

复核记录位于 `results/state-wm-video-20261011/audit.json`、`metric-comparison.json`。本地完整运行副本位于：

`K:/world-model/deployments/wm-safetylab/results/state-wm-video-20261011/run_d57f3484d13949ec/`

独立评价完整副本位于同级 `run_d37d3bf23a1544a9/`。可离线打开模型运行 `output/visualization.html`，视频、图片和 CSV 均使用相对路径。

## 实验设置

- 原实验：`run_b474c67a396441c2`，不含录像，保留原位。
- 本次新运行：`run_d57f3484d13949ec`，重新采集、训练和预测；没有把旧预测叠到新录像上。
- 场景：ManiSkill 3.0.1、Panda、PushCube-v1；`state` 输入、`pd_joint_pos` 控制、CPU 物理仿真与 GPU 渲染。
- 固定官方规划器，七个关节目标加入标准差 0.003 rad 的独立高斯扰动。模型输入实际限幅执行动作。
- 36 个回合；24 个训练、6 个验证、6 个测试；数据种子 0—35，划分种子 20261010。
- 两层 128 隐单元残差 MLP；动作条件模型与无动作模型各训练 seed 0、1、2，最多 400 epochs，验证集选择 checkpoint。
- 录像覆盖全部测试回合：3、4、17、18、21、35；按编号顺序播放，没有按测试表现挑选。
- 可视化固定训练种子 0；独立评价仍按回合等权、再对三个训练种子平均。

## 查看产物

历史部署的本机入口（仅在对应服务及 SSH 隧道在线、运行记录仍存在时可用）：

[中文展示页](http://127.0.0.1:18765/api/v1/runs/run_d57f3484d13949ec/files/visualization.html)

| 文件 | 用途 |
|---|---|
| `simulation.mp4` | 全部测试回合仿真录像合集；可见机械臂、方块和目标区域 |
| `comparison.mp4` | 同步仿真、方块 XYZ、末端 XYZ、三维欧氏误差；5 步预测，1600×1000 |
| `videos/episode_*.mp4` | 各回合原始 512×512 录像 |
| `predictions_h1_seed0.csv` | 一步预测完整逐帧数据，含实际/预测位置、误差、预测起点和目标帧 |
| `predictions_h5_seed0.csv` | 五步预测完整逐帧数据 |
| `charts/episode_*_h*.png/.pdf` | 全部测试回合的一步、五步定量曲线；PNG 300 DPI、PDF 向量格式 |
| `recordings.json` | 摄像机、帧数、每回合文件、目标位置、帧与状态的对应关系 |
| `visualization.json` | 合集帧偏移、预测起点与目标帧映射、训练种子和展示范围 |
| `comparison_preview.png` | 第一测试回合中间帧预览；按固定位置选择 |

颜色：黑色为真实状态，蓝色为动作条件 WM，橙色为无动作 WM，绿色为状态保持基线。位置和三维欧氏误差统一使用 mm；XYZ 为世界坐标。

## 时间对齐规则

控制频率和视频帧率均为 20 Hz，每个回合保存初始帧，再保存每个控制步执行后的帧。因此 N 条状态转移对应 N+1 帧。

对目标帧 k，H 步预测从真实状态 k−H 初始化，然后使用该时间窗口内记录的动作递归预测。状态保持基线使用同一个起点状态，不使用目标时刻真值。H=5 对应 0.25 秒。

最初 H 帧没有预测，图中保持空缺，CSV 的 `prediction_available` 为 0。时间游标与视频帧同步，曲线展示整个测试回合，属于事后分析，不是在线界面。录像展示单个训练种子，不能将它与三个种子的平均指标视为完全相同的统计量。

## 下次如何运行

无需增加模型包、服务或网页模块，现有结果页会自动识别 MP4 和 PNG。视频采集与绘图依赖位于已有 ManiSkill 原生环境，平台核心没有增加 NumPy、Matplotlib 或 Torch 依赖。

可在网页实验配置的 `inputs` 中加入：

```json
{"record_video": true, "visualization_seed": 0}
```

CLI 示例（Windows，本地隧道已经连接到服务器时）：

```powershell
. K:\world-model\use-k-drive.ps1
Set-Location K:\world-model\deployments\wm-safetylab
$env:WMS_BASE_URL = 'http://127.0.0.1:18765'
.\.venv\Scripts\python.exe -m wms run examples/state-wm-maniskill-video.json --wait
```

Python 入口仍为 `wms.run`，例如：

```python
import os
import wms
os.environ['WMS_BASE_URL'] = 'http://127.0.0.1:18765'
run = wms.run({'recipe': 'state-wm-maniskill',
               'inputs': {'record_video': True, 'visualization_seed': 0}})
result = wms.wait(run['id'])
```

不启用 `record_video` 时保留原执行路径。录制要求重新运行仿真，不允许同时指定 `dataset_dir`，因为过去未保存的画面无法从状态文件直接还原。

## 修改范围

| 模块 | 改动 |
|---|---|
| `src/wms/engines/state_wm_maniskill/collect.py` | 提前计算原有数据划分；只对全部测试回合录像，记录初始帧与每步后帧 |
| `src/wms/engines/state_wm_maniskill/visualize.py` | 独立可视化模块：帧对齐、CSV、定量曲线、视频合成和中文报告 |
| `src/wms/engines/state_wm_maniskill/native.py` | 开关控制可视化调用，将新增产物登记到运行结果 |
| `native/state_wm_maniskill/recipes.json` | 增加可选 `record_video` 与 `visualization_seed` 默认参数 |
| `tests/test_state_wm_visualization.py` | 数值测试覆盖一步/多步对齐、状态保持、训练种子、数据来源与缺失时间步 |

服务器只更新此实验模块；原有环境、权重、机器配置和旧运行未替换。服务核心、API 和前端代码没有修改。原生工作进程会加载新代码，运行时可直接传入上述参数。

## 结论边界

视频来自真实仿真渲染，不是 WM 生成的视频。它展示预测后果是否贴近真实后果，不证明概率校准、安全防护效果、反馈更新效果，也不是论文复现。
