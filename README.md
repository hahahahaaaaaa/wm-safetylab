# WM-SafetyLab

面向世界模型研究的 Python 核心、CLI 与 Web 实验工作台。当前版本 **0.3.0**。

```text
实验配方 → 独立 Python 工作进程 → 原始输出 / checkpoint / 轨迹 / 视频
                                  ↓
                            独立评价任务 → 结果比较
```

支持视频生成、状态/语言/视频世界模型与行动研究。四层研究职责为 **数据 S1 → 模型 S2/S3 → 算法 S4 → 执行与反馈 S5/S6**；失效、攻击、防护和评测通过具体实验配方与指标实现。接口本身不证明安全性，也不要求每个模型都有规划器。

## 文档入口

| 目标 | 文档 |
|---|---|
| 了解平台有哪些模型、能做什么、如何实现及未来扩展 | [项目说明](docs/PROJECT_OVERVIEW.zh-CN.md) |
| 从新机器部署到第一次实验 | [部署与日常使用](docs/DEPLOYMENT_AND_USAGE.zh-CN.md) |
| 每个模型的安装、配方、参数、输出和验证范围 | [全部模型与实验使用手册](docs/MODELS_AND_EXPERIMENTS.zh-CN.md) |
| 后端、CLI、Python API 和独立评价 | [开发与接口手册](docs/DEVELOPMENT.zh-CN.md) |
| 添加新模型或仿真器 | [开发总指南](docs/ONBOARDING_MODELS_AND_ENVIRONMENTS.zh-CN.md)与[模板](templates/README.zh-CN.md) |
| 真实模型与机械臂接入 | [TD-MPC2 × PushCube 教程](docs/TD_MPC2_PUSHCUBE_GUIDE.zh-CN.md) |
| TD-MPC2 实际实验结果 | [验收报告](docs/TD_MPC2_PUSHCUBE_ACCEPTANCE_20261012.zh-CN.md) |
| 上游项目与发布边界 | [第三方来源](THIRD_PARTY.md) |

## 快速开始

安装 Git、Python 3.11、Node.js 22。先只安装平台控制环境，模型依赖按需安装到各自环境。

Linux：

```bash
git clone https://github.com/hahahahaaaaaa/wm-safetylab.git
cd wm-safetylab
source scripts/linux/use-linux.sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
(cd web && npm ci && npm run build)
.venv/bin/python -m wms serve --host 127.0.0.1 --port 8765
```

Windows PowerShell（先进入你选定的项目存储目录）：

```powershell
git clone https://github.com/hahahahaaaaaa/wm-safetylab.git
Set-Location wm-safetylab
. .\scripts\windows\use-workspace.ps1
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
Push-Location web
npm ci
npm run build
Pop-Location
.\.venv\Scripts\python.exe -m wms serve --host 127.0.0.1 --port 8765
```

打开 <http://127.0.0.1:8765/>；API 文档位于 <http://127.0.0.1:8765/docs>。第二个终端使用相同虚拟环境的 Python：

```bash
python -m wms recipes
python -m wms run demo-rollout --wait
python -m wms runs
```

上述第二终端需要先激活 `.venv`，或直接使用其 Python 的完整路径。这是 CPU 合成示例，不代表真实世界模型能力。模型完成后自动评价可能还在队列中。停止前台服务使用 Ctrl+C。

## 已有接入

| 类型 | 当前代码入口 | 使用边界 |
|---|---|---|
| 视频 | Wan2.1、ModelScope | 文本到视频生成，不能直接当作控制状态 |
| 潜在状态模型 | DreamerV3 | 原生 checkpoint 预测；不等同于完整在线搜索 |
| 语言/视频 WM | RLVR-World 语言与视频实例 | 原生格式预测；不含自动机械臂闭环或 RLVR 全量后训练 |
| 反馈更新研究 | RENEW 2×2 2048 | 受控反馈与更新机制实验，非完整论文复现 |
| 行动 | OpenVLA | 图像与指令到动作；当前不提供 LIBERO 全量基准 |
| 仿真 | ManiSkill PushCube、Pendulum 教学模板 | 环境运行、轨迹与录像；安全信号按实际实现报告 |
| 状态 WM | 小型状态 WM＋ManiSkill | 采集、训练、后果预测与物理位置对照 |
| 可训练控制 WM | TD-MPC2＋PushCube | 已实跑训练、MPPI 控制、动作扰动及潜空间预测 |

GitHub 不包含虚拟环境、模型权重、数据集、机器配置或完整实验录像。首次克隆时未配置的原生配方会显示未配置；按使用手册准备资源和 runtime 文件即可启用。每个模型的真实验证范围见模型手册，不能以配方存在代替运行成功。

## Python 使用

```python
import wms

run = wms.run({"recipe": "demo-rollout", "seed": 0, "timeout_s": 60})
finished = wms.wait(run["id"])
if finished.get("automatic_evaluation_id"):
    evaluation = wms.wait(finished["automatic_evaluation_id"])
    print(evaluation["result"])
print(wms.compare([run["id"]]))
```

客户端默认连接本机 8765；通过环境变量 `WMS_BASE_URL` 选择其他服务，例如 SSH 隧道端口。原生环境中的 worker 不需要启动第二个 API 服务。

## 代码组织

```text
src/wms/app/               HTTP API 与 CLI
src/wms/core/              配方、运行队列、存储、评价、组件协议
src/wms/runners/unified.py generate / predict / rollout / plan / native / evaluate
src/wms/engines/           各模型原生入口、兼容层与指标
src/wms/scenarios/         仿真环境适配
src/wms/research/          反馈与更新研究函数
native/<engine>/          共享 recipes、配置示例与资源准备工具
examples/                 CPU 与真实实验客户端示例
templates/                新模型和新环境教学模板
web/                      React 工作台
docs/                     当前使用与开发文档
```

新增模型通常只新增自己的代码和 `native/<engine>/recipes.json`。默认自动发现配方，也支持 `--registry` 加载外部注册表。

当前为单服务、单队列、单机执行；不同任务可选不同解释器，同一组合任务的模型与环境仍需共用可兼容的解释器。默认绑定回环地址，团队远程使用可通过 SSH 隧道。服务没有多租户认证与权限隔离，不应直接暴露到公共互联网。

## 开发检查

```bash
python -m pytest tests
cd web
npm run build
```

GitHub Actions 包含 Windows/Linux CPU 测试与 Web 构建。GPU 验收需要目标机器上的原生依赖、数据和权重，CPU CI 不代替真实模型验证。
