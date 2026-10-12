# TD-MPC2 + ManiSkill PushCube 接入计划与过程记录

开始日期：2026-10-12。目标：通过 WM-SafetyLab 的真实服务任务完成动作条件潜空间世界模型训练、检查点回载、独立闭环评测、潜空间预测与仿真对照、录像，并提供能复用的接入教程。

## 实施边界

- 采用 ManiSkill 官方仓库 `examples/baselines/tdmpc2`；记录实际 Git 提交。上游源码保持干净，兼容调整在 `src/wms/engines/tdmpc2/`。
- 首先锁定 PushCube-v1、Panda、state 输入和官方基线控制模式 pd_joint_delta_pos；不与已有 pd_joint_pos 轨迹混用。
- 保留现有服务数据、机器配置、模型权重、原生环境；只新增本模型文件和独立运行环境。
- 真实训练成功与达到论文性能分开记录；单次种子验收不能作为安全效果或论文复现结论。
- TD-MPC2 预测的是潜状态、奖励和价值，不伪装为物理坐标或已校准安全概率。

## 顺序及验收

| 阶段 | 工作 | 验收证据 |
|---|---|---|
| 1 | 核对服务器、官方代码、已安装 ManiSkill/PyTorch | 机器资源、上游提交、环境清单 |
| 2 | 独立依赖环境，最小兼容模块 | 真环境 reset/step、模型 forward/update、权重保存/加载 |
| 3 | 编写平台原生训练、评测、预测入口与配方 | 平台 API 提交，真实 worker 使用指定解释器 |
| 4 | 真实训练并保存模型、数据/配置来源和训练曲线 | run ID、训练步数与更新数、checkpoint、损失记录 |
| 5 | 独立评测与固定动作预测；录像和独立指标任务 | 成功/回报、预测误差、视频、来源运行与评价 ID |
| 6 | 整理中间兼容函数、逐步命令和问题修复过程 | 中文教程、接口说明、锁定依赖与复验配置 |

## 分工

- 主执行：部署、原生封装、正式任务运行及服务集成。
- 并行代码审阅：官方模型 API、训练步数和潜空间输出语义。
- 并行文档审阅：与平台 native/metrics/runtime 接口对应关系。

2026-10-12 验收完成。实际执行日志见 `results/tdmpc2-integration-20261012/PROCESS_LOG.zh-CN.md`，机器可读证据见同目录 `acceptance/acceptance.json`；开发教程见 `docs/TD_MPC2_PUSHCUBE_GUIDE.zh-CN.md`，最终结论见 `docs/TD_MPC2_PUSHCUBE_ACCEPTANCE_20261012.zh-CN.md`。
