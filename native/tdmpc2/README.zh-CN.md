# TD-MPC2 × ManiSkill PushCube

2026-10-12 已在平台完成真实 GPU 训练、评测、动作扰动、预测、录像和独立评分。

- [完整接入教程](../../docs/TD_MPC2_PUSHCUBE_GUIDE.zh-CN.md)：环境、状态/动作兼容函数、回放时序、任务入口和扩展方法。
- [正式验收报告](../../docs/TD_MPC2_PUSHCUBE_ACCEPTANCE_20261012.zh-CN.md)：预算、结果、运行 ID 与结论边界。
- [全过程](../../docs/TD_MPC2_PROCESS_LOG.zh-CN.md)：实际安装、失败与修复记录；原始机器文件和运行工件保留在本地。
- [全流程客户端](../../examples/tdmpc2/run_study.py)：从头运行，或通过 `--training-run` 复用已完成模型。
- [通用组件客户端](../../examples/tdmpc2/use_existing_baseline.py)：通过相同平台的 predict/rollout 使用模型。

机器配置 `runtime.json` 只在服务器保存；本地 `runtime.linux.json` 是该服务器配置示例，`runtime.example.json` 用于新机器填写。已有平台服务自动发现 `recipes.json` 中的三条配方。更新数值参数无需重启；首次发现新增配方才需服务重新加载。

```powershell
. K:/world-model/use-k-drive.ps1
Set-Location K:/world-model/deployments/wm-safetylab
$env:WMS_BASE_URL = 'http://127.0.0.1:18765'

# 使用本次训练的模型，进行新的三步预测与真实仿真对照
.venv/Scripts/python.exe examples/tdmpc2/use_existing_baseline.py run_d5fc89b69fb84542 --task predict

# 复用本次训练模型，新建三组评测与固定动作预测
.venv/Scripts/python.exe examples/tdmpc2/run_study.py --training-run run_d5fc89b69fb84542
```

本次工程基线训练 seed 0、102,400 transitions；留出 10 种子下初始模型 0/10、训练后 10/10、动作噪声 0.1 组 10/10。它不代表完整论文复现或安全性认证。TD-MPC2 的预测输出是 latent/reward，不是已解码的物理位置或校准安全概率。
