# 部署与日常使用

本文面向首次克隆 WM-SafetyLab 的使用者，按当前 **0.3.0** 实现说明安装、启动、首次实验、原生环境配置和结果检查。配置与开发接口见 [开发手册](DEVELOPMENT.zh-CN.md)，新模型与仿真器接入见 [接入总指南](ONBOARDING_MODELS_AND_ENVIRONMENTS.zh-CN.md)。

## 1. 先区分两个 Python 环境

| 环境 | 用途 | 需要安装什么 |
|---|---|---|
| 控制环境 `.venv` | HTTP 服务、Web 后端、CLI、Python 客户端、CPU 接口示例 | 本仓库 `pyproject.toml` 中的依赖 |
| 原生工作环境 | 执行指定模型、仿真器或评价函数 | 该组件需要的 PyTorch/JAX、CUDA 相关库、仿真器、模型包等 |

服务按每次实验的 `runtime.python` 启动工作进程。不同实验可以使用不同环境；同一个普通 `rollout`、`predict` 或 `plan` 任务中的模型、环境和规划器共用一个解释器，必须能在该环境内同时导入。独立评价也可选择自己的解释器。

安装控制环境不会安装所有原生模型，也不会下载权重或仿真资源。先跑通 CPU 示例，再按需要安装一个原生环境。不要把互相冲突的原生依赖全部装进 `.venv`。

前置工具为 Git、**Python 3.11** 和带 npm 的 **Node.js 22**。项目允许 Python 3.11 及以上；目前 CI 使用 Python 3.11，Web 构建使用 Node.js 22。GPU 型号、显存、驱动和系统要求取决于实际模型，CPU 示例不需要 GPU。

## 2. 克隆与安装

以下使用自选存储根目录；把占位路径换成你有写权限的实际位置。已有仓库时进入原目录，跳过克隆，不覆盖已有环境、权重和实验数据。原生源码、缓存、环境、模型资源也应放在该存储根目录下。

### Windows PowerShell

```powershell
$WorkRoot = Read-Host '输入项目存储根目录的绝对路径'
$ProjectRoot = Join-Path $WorkRoot 'deployments\wm-safetylab'
New-Item -ItemType Directory -Force -Path (Split-Path $ProjectRoot) | Out-Null
git clone https://github.com/hahahahaaaaaa/wm-safetylab.git $ProjectRoot
Set-Location -LiteralPath $ProjectRoot

# 每个安装或运行项目代码的新终端都先执行环境脚本。
$env:WMS_WORKSPACE_ROOT = $WorkRoot
. .\scripts\windows\use-workspace.ps1

py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -e .
Push-Location web
npm ci
npm run build
Pop-Location
```

如果没有 Windows `py` 启动器，用已安装的 Python 3.11 可执行文件代替 `py -3.11`。上面的安装命令显式选择 `.venv` 中的 Python，无需先激活环境。

环境脚本将临时目录和常用依赖缓存放在选定工作空间中。以后新开 PowerShell，进入项目目录后重新设置 `WMS_WORKSPACE_ROOT` 并点加载脚本；已有专用存储初始化脚本的部署，应继续遵守该部署的存储规则。

### Linux Bash

```bash
read -r -p '输入项目存储根目录的绝对路径: ' WORK_ROOT
PROJECT_ROOT="$WORK_ROOT/deployments/wm-safetylab"
mkdir -p "$(dirname "$PROJECT_ROOT")"
git clone https://github.com/hahahahaaaaaa/wm-safetylab.git "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

export WMS_ROOT="$WORK_ROOT"
source scripts/linux/use-linux.sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
(cd web && npm ci && npm run build)
```

Linux 发行版需提供可用的 Python 3.11 `venv` 模块。以后新开 Bash，进入项目目录后重新设置 `WMS_ROOT` 并加载环境脚本。`scripts/linux/use-linux.sh` 会设置临时目录、pip/npm、Hugging Face、PyTorch 等缓存位置；它不会安装 CUDA 或模型依赖。

`npm ci` 按 `web/package-lock.json` 安装依赖；`npm run build` 先运行 TypeScript 编译，再生成 `web/dist`。后端启动时发现 `web/dist/index.html` 才会提供网页，因此首次使用需要先完成构建。

开发测试可在控制环境额外安装 `.[dev]`，需要本包提供的 FFmpeg 辅助时可安装 `.[video]`。具体原生组件的依赖仍按该组件说明安装。

## 3. 启动、连接与停止

从项目根目录启动；保持当前终端打开。

Windows：

```powershell
. .\scripts\windows\use-workspace.ps1
& .\.venv\Scripts\python.exe -m wms serve `
  --host 127.0.0.1 --port 8765 --storage-root .\data
```

Linux：

```bash
source scripts/linux/use-linux.sh
.venv/bin/python -m wms serve \
  --host 127.0.0.1 --port 8765 --storage-root ./data
```

访问 [Web 工作台](http://127.0.0.1:8765/)、[交互式 API 文档](http://127.0.0.1:8765/docs) 或 [健康检查](http://127.0.0.1:8765/api/v1/health)。健康检查应返回 `status: "ok"` 与 `version: "0.3.0"`。

仓库也提供启动包装：Windows 使用 `./start.ps1`；Linux 使用 `bash scripts/linux/serve.sh`。Linux 包装默认使用 `data-linux`，可通过 `WMS_STORAGE_ROOT` 修改；上面的直接命令明确使用 `data`。切换启动方式时，保持同一个实际存储目录才能看到原来的运行。

| 启动选项 | 用途 |
|---|---|
| `--storage-root PATH` | 保存数据库、运行配置、日志和产物；未指定时读 `WMS_STORAGE_ROOT`，否则使用项目 `data` |
| `--worker-python PATH` | 没有显式 `runtime.python` 时使用的默认工作解释器；默认是服务的 Python |
| `--registry PATH` | 额外加载组件与配方文件，可重复；`--components` 是同一选项的别名 |
| `--native-configs PATH` | 读取原生环境名称到机器配置文件的 JSON 映射 |
| `--host`、`--port` | 监听地址、端口；默认 `127.0.0.1:8765` |

**同一个存储目录只运行一个服务实例。** 当前是单机 FIFO 队列，同一时刻执行一个实验或评价任务；不要为同一存储目录另开服务或多个 Uvicorn worker。另开测试服务时使用不同端口和不同存储目录。

前台服务按 `Ctrl+C` 停止。停止前先等待运行结束，或在 Web/CLI 中取消任务；正常退出会停止剩余队列和工作进程。异常退出后，下一次启动会把此前未结束的记录标为 `interrupted`，需要显式重跑，不会自动续训。`stop.ps1` 只针对它能够识别的、已记录 PID 的后台实例；它不是前台启动命令的必需步骤。

### 从其他终端或远程客户端连接

CLI 和 Python 客户端使用 `WMS_BASE_URL`，值是服务地址，**不要附加 `/api/v1`**。

```powershell
$env:WMS_BASE_URL = 'http://127.0.0.1:8765'
```

```bash
export WMS_BASE_URL=http://127.0.0.1:8765
```

远程服务器推荐保留回环监听，通过你自己的 SSH 连接转发端口：

```bash
ssh -N -L 18765:127.0.0.1:8765 YOUR_SSH_HOST_ALIAS
```

此时本地浏览器访问 [转发后的工作台](http://127.0.0.1:18765/)，本地客户端设置 `WMS_BASE_URL=http://127.0.0.1:18765`。替换 `YOUR_SSH_HOST_ALIAS` 为你的 SSH 配置别名。服务当前没有用户认证功能，且运行配置可以指定 Python 入口和环境，不应直接公开到互联网。

客户端提交的是 JSON；API 不会上传源码、权重或把客户端路径转换为服务器路径。所有 `runtime`、checkpoint、输入文件路径都必须能被**服务所在机器**访问。

## 4. 第一次实验：CPU 完整流程

服务启动后，在另一个已加载环境脚本的终端、项目根目录执行。

Windows：

```powershell
& .\.venv\Scripts\python.exe -m wms recipes
& .\.venv\Scripts\python.exe -m wms components
& .\.venv\Scripts\python.exe -m wms run demo-rollout --wait
```

Linux：

```bash
.venv/bin/python -m wms recipes
.venv/bin/python -m wms components
.venv/bin/python -m wms run demo-rollout --wait
```

最后一条输出一个运行记录，记下 `id`。预期模型 `state` 为 `completed`，原始输出包含 `trajectory.json`。该配方含指标，因此记录还会包含 `automatic_evaluation_id`：这是随后排队的独立评价，**`run --wait` 只等待源实验，不等待自动评价**。在网页“运行”或“结果”中等待评价完成，可看到 `steps`、`return`、`cost_sum`。

其他无需 GPU 的内置示例：

| 配方 | 演示内容 |
|---|---|
| `demo-predict` | 离线线性状态预测 |
| `demo-plan` | 预测模型、采样规划器与环境交互 |
| `demo-generate` | 生成一个文本文件，不生成视频 |
| `demo-native` | 调用一个原生函数并登记 JSON 文件 |

这些是合成接口示例，用于验证安装与任务流程，不代表真实世界模型性能或安全效果。

### 修改参数与使用 Python

后面的简短示例使用 `python -m wms`。在当前客户端终端先激活**控制环境**，确保这些命令调用正确的解释器：

```powershell
# Windows PowerShell
. .\.venv\Scripts\Activate.ps1
```

```bash
# Linux Bash
source .venv/bin/activate
```

也可继续把 `python` 替换为完整的 `.venv` 解释器路径，不激活环境。

把下面 JSON 保存为项目内自建的 `runtime_files/first-run.json`，先创建 `runtime_files` 目录。它是单次实验配置，不是注册表：

```json
{
  "recipe": "demo-rollout",
  "name": "first CPU experiment",
  "seed": 7,
  "steps": 12,
  "timeout_s": 60
}
```

执行 `python -m wms run runtime_files/first-run.json --wait`。

也可以将下列代码保存为自己的 Python 脚本，再用控制环境运行。它演示等待源实验、等待自动评价，以及指定评价来源进行比较：

```python
import wms

record = wms.run({"recipe": "demo-rollout", "seed": 7, "timeout_s": 60})
source = wms.wait(record["id"], timeout=120)
if source["state"] != "completed":
    raise RuntimeError(source.get("error"))

score = wms.wait(source["automatic_evaluation_id"], timeout=120)
if score["state"] != "completed" or score.get("evaluation_status") != "completed":
    raise RuntimeError(score.get("error") or score.get("metric_errors"))

print(score["result"]["metrics"])
print(wms.compare([source["id"]], evaluations={source["id"]: score["id"]}))
```

`timeout_s` 是服务端工作进程执行时限，不包含排队时间；`wms.wait(..., timeout=...)` 是客户端等待时限。后者超时不会自动取消服务端任务。长训练请设置足够的客户端等待时间，或只提交后在网页查看。

## 5. 原生模型与多环境配置

### 机器文件放在哪里

本文约定用 `runtime_files/` 保存本机填写的配置。**它只是目录名，不是请求中的字段，也不会被自动扫描。** 共享配方和适配器可以纳入版本控制，实际机器配置、权重、环境和私有输入应按你的部署流程单独管理。

```text
wm-safetylab/
├── .venv/                         控制环境
├── native/<model>/recipes.json    共享配方
├── runtime_files/
│   ├── native-configs.json        原生环境映射
│   ├── my-model.local.json        本机解释器与资源
│   └── first-run.json             单次实验覆盖项
└── data/                          服务端持久化数据
```

先按模型自身说明安装、验证原生环境，准备权重和输入。上游源码保持干净；研究适配器放在本平台 `src/wms/engines/<model>/`、`src/wms/scenarios/<environment>/` 或独立研究包中。服务启动工作进程时会将本仓库 `src` 加入 `PYTHONPATH`，因此通过服务执行通常无需往原生环境再安装整套控制依赖；原生组件自身需要的包仍必须安装。

### 填写实际 runtime

通用机器配置的结构如下，**必须替换路径**；JSON 不会展开 `$变量`、`${变量}` 或 `~`：

```json
{
  "python": "/YOUR_STORAGE_ROOT/envs/my-model/bin/python",
  "python_paths": ["/YOUR_STORAGE_ROOT/deployments/my-model-adapter"],
  "env": {"CUDA_VISIBLE_DEVICES": "0"},
  "timeout_s": 3600
}
```

Windows 的 `python` 改为原生环境中的 `Scripts/python.exe`，JSON 路径可使用正斜杠。CPU 环境可省略 `CUDA_VISIBLE_DEVICES`；依赖 GPU 的环境仍需满足实际驱动与包版本要求。

Registry 从机器文件中提取 `python`、`python_paths`、`cwd`、`env`、`timeout_s`；checkpoint、源码、仿真资源等字段由各原生入口自己读取。因此通用配置不能替代模型专用模板：例如 [TD-MPC2 模板](../native/tdmpc2/runtime.example.json)、[Wan 模板](../native/wan/runtime.example.json)。检查对应原生入口真正需要的字段，填成服务端可访问的路径，不要直接使用模板占位值。

### 用映射切换本机配置

例如先准备好 `runtime_files/tdmpc2.local.json` 和 `runtime_files/maniskill.local.json`，再创建 `runtime_files/native-configs.json`：

```json
{
  "tdmpc2": "tdmpc2.local.json",
  "maniskill": "maniskill.local.json"
}
```

启动服务时增加：

```text
--native-configs runtime_files/native-configs.json
```

映射中的键要与配方 `runtime.native_key` 一致；相对文件路径以**映射 JSON 所在目录**为起点。无需编辑共享配方中的默认机器路径。启动后用 `recipes` 检查 `configured` 和 `configuration_error`；`configured=true` 只表示配置解析成功，没有检查依赖、权重完整性或 GPU 可用性。

也可以在单次实验中选择文件，不改服务默认映射：

```json
{
  "recipe": "tdmpc2-pushcube-eval",
  "runtime": {"config_file": "runtime_files/tdmpc2.local.json"},
  "inputs": {"checkpoint": "/YOUR_STORAGE_ROOT/weights/your-checkpoint.pt"},
  "params": {"episodes": 2, "video_episodes": 1}
}
```

此例要求你已经安装 TD-MPC2 所需环境并拥有对应任务的真实 checkpoint，不是首次安装后的 CPU 命令。更小的首次原生验证应选择该模型提供的 smoke 配方或最小输入。

配置选择规则：请求 `runtime.config_file` 优先于启动映射，映射优先于配方默认文件；运行设置按“机器文件 → 配方显式 runtime → 请求显式 runtime”合并，后者覆盖前者，`env` 按键合并。服务默认解释器只在 `runtime.python` 缺省或为 `auto` 时使用。

路径基准也要区分：注册表里的相对运行路径以注册表所在目录为基准；机器文件中的相对运行路径以机器文件为基准；HTTP/CLI 提交的内联 `runtime` 相对路径以**服务端项目根目录**为基准，不以你保存单次实验 JSON 的位置为基准。模型专有资源字段是否支持相对路径取决于适配器，优先填写绝对路径。

同一队列可以依次运行环境 A、环境 B 和评价环境 C。若模型与仿真器依赖冲突，先采用“采集文件 → 离线预测/训练 → 独立评价”的方式，或自己实现明确的跨进程适配；平台不会自动跨虚拟环境拼接一个闭环任务。

## 6. 日常运行、独立评价与对比

Web 有四个页面：

| 页面 | 用途 |
|---|---|
| 实验 | 选择配方、编辑参数、运行环境与执行超时，提交任务 |
| 运行 | 查看实验和评价的排队/执行状态、日志，取消或重跑 |
| 结果 | 查看原始文件、下载工件、查看独立评价历史和错误 |
| 对比 | 选择来源实验及其评价记录，查看指标和可比性提示 |

以下命令中的 `RUN_ID` 替换为真实来源实验 ID：

```bash
python -m wms runs
python -m wms get-run RUN_ID
python -m wms logs RUN_ID
python -m wms cancel RUN_ID
python -m wms retry RUN_ID
python -m wms evaluate RUN_ID --wait --timeout-s 120
python -m wms compare RUN_ID_A RUN_ID_B
```

按需要分别执行，不要把整段当作顺序操作：取消会停止该任务；重跑只允许已结束的任务，并创建新 ID，保留旧记录。重跑沿用已解析的配置，不会自动改成后来修改的配方默认值。

评价要求来源实验已完成且原始 `result.json` 存在。它创建新的 `task: "evaluate"` 记录，以 `source_run` 关联原实验；评价失败不改写原模型的完成状态。`metrics: []` 的实验不自动评价；未指定 `--metrics` 时，手动评价沿用来源配置中的指标列表。

自定义指标文件 `runtime_files/metrics.json` 是 JSON **数组**：

```json
[
  {
    "entrypoint": "wms_demo:trajectory_metrics",
    "name": "trajectory",
    "params": {},
    "version": "demo-1"
  }
]
```

它适用于上述 CPU 轨迹示例。不要把不读取相同数据格式的指标套在其他模型上。执行：

```bash
python -m wms evaluate RUN_ID --metrics runtime_files/metrics.json --wait
```

需要换评价环境时，`--runtime` 接收一个 JSON 对象文件，例如内容为 `{"config_file": "runtime_files/evaluator.local.json"}`，或直接包含 `python`、`python_paths`、`env`；它不是单独的解释器字符串。评价继承源实验运行设置，再应用覆盖项。

评价中某些指标异常时，运行本身可能是 `completed`，同时 `evaluation_status` 为 `partial_failed`，错误在 `metric_errors`。`evaluate --wait` 对这种情况返回非零退出码。查看具体错误，不把缺失指标当成零，也不从旧评价补值。

默认比较会为每个实验选择最近完成的评价；如果要固定来源，创建 `runtime_files/evaluations.json`：

```json
{
  "RUN_ID_A": "EVALUATION_ID_A",
  "RUN_ID_B": "EVALUATION_ID_B"
}
```

```bash
python -m wms compare RUN_ID_A RUN_ID_B --evaluations runtime_files/evaluations.json
```

这里的键和值都换成真实 ID，且评价必须属于对应来源实验。返回数据会标明来源 ID、时间、指标入口、参数和版本；不同指标定义、任务、输入或数据集不能仅凭数值高低排名。原生指标与独立评价分开展示。

## 7. 日志、原始文件和视频

每个存储根目录的当前布局如下：

```text
data/
├── runs.sqlite3
└── runs/<run_id>/
    ├── config.json           本次实际执行配置
    ├── run.log               工作进程标准输出和错误
    └── output/
        ├── result.json       该实验/评价的结果
        └── ...               轨迹、预测、checkpoint、视频等
```

独立评价有自己的 `<run_id>`、日志和 `output/metrics.json`，不会覆盖源实验目录。失败时先看记录的 `error`，再查看该 ID 的日志和实际 `config.json`，确认运行的解释器、输入与所选配置文件。

当前 HTTP 路由统一位于 `/api/v1`。以下是日常查询入口，完整请求结构以正在运行的 `/docs` 为准：

| 方法与路径 | 返回内容 |
|---|---|
| `GET /api/v1/health` | 服务状态、版本 |
| `GET /api/v1/recipes` | 可发现的配方与配置状态 |
| `GET /api/v1/components` | 已注册组件 |
| `POST /api/v1/runs` | 请求体为实验配置，提交新任务 |
| `GET /api/v1/runs/{id}` | 运行记录与配置、结果 |
| `GET /api/v1/runs/{id}/logs?offset=0` | 日志片段及 `next_offset` |
| `GET /api/v1/runs/{id}/result` | 原始结果对象 |
| `GET /api/v1/runs/{id}/files` | 已登记工件清单 |
| `GET /api/v1/runs/{id}/files/{path}` | 下载该运行输出目录中的文件 |
| `GET /api/v1/runs/{id}/evaluations` | 此来源实验的独立评价历史 |
| `POST /api/v1/runs/{id}/evaluate` | 创建评价；可传 `metrics`、`runtime`、`timeout_s` |
| `POST /api/v1/comparisons` | 请求体包含 `run_ids` 和可选的 `evaluations` 映射 |

日志每次最多返回 200,000 个字符，`logs` 命令读取首段；长日志可查看服务器的 `run.log`，或用响应的 `next_offset` 继续请求。控制台服务日志与任务 `run.log` 是两类日志，模型异常主要在后者。

视频只有在实际组件生成并登记输出文件后才可展示。原生入口应返回类似：

```json
{"outputs": [{"kind": "video", "path": "simulation.mp4", "media_type": "video/mp4"}]}
```

文件必须真实存在于本次 `output_dir` 内。网页“结果”页可以播放已登记的视频或下载原文件；浏览器不支持某种编码时，下载后用兼容播放器检查。`demo-generate` 只生成文本，普通 `rollout` 只自动登记轨迹，不会自动为任何仿真器录像。真实录像需要相应渲染依赖与适配入口，例如 [环境模板](../templates/environment/README.zh-CN.md) 提供的单摆录像配方。

## 8. 添加自己的模型或环境

服务默认加载 `examples/minimal/registry.json` 和 `native/*/recipes.json`，再加载启动时指定的 `--registry` 文件。推荐先复制 [模型模板](../templates/model/README.zh-CN.md) 或 [环境模板](../templates/environment/README.zh-CN.md)，在自己的目录中修改组件包名与配方 ID。

最小接口和文件位置：

| 接入对象 | 入口要求 | 通常放置位置 |
|---|---|---|
| 预测/生成/策略模型 | `Class(config)`，按任务实现 `predict(request)`、`generate(request)` 或 `act(obs, instruction)` | `src/wms/engines/<model>/` 或独立 Python 包 |
| 仿真环境 | `Class(config)`，实现 `reset(seed, inputs)`、`step(action)`、`close()` | `src/wms/scenarios/<environment>/` 或独立 Python 包 |
| 原生训练或论文流程 | `function(config, output_dir, context)` | 自己的 `native.py` |
| 指标 | `function(source_result, params)`，读取源产物，写入评价目录 | 自己的 `metrics.py` |
| 共享配方 | 唯一 ID、task、entrypoint/组件、默认 inputs/params | `native/<name>/recipes.json` 或外部注册表 |
| 机器配置 | 解释器、资源路径和环境变量 | `runtime_files/*.local.json` |

外部注册表通过顶层 `python_paths` 提供组件导入路径，再用以下启动选项加载：

```text
--registry path/to/your/registry.json
```

修改注册表后重启服务才会刷新配方列表；每次实验的参数覆盖不需要重启。同组重复 ID 会报错，不能用加载顺序覆盖已有组件。模型与环境的观测形状、动作单位、时间尺度仍须明确匹配，函数签名相同不代表可直接配对。

先在原生环境跑通最小官方示例，再接一条小规模配方、检查原始输出，最后增加独立评价。教学模板中的占位真实后端需要由接入者实现，不应把教学输出当作真实模型结果。详细协议、原生包装、规划与评价开发见 [开发手册](DEVELOPMENT.zh-CN.md)。

## 9. 更新、备份与常见问题

更新代码前等待任务结束并停止服务，记录当前版本；更新后在控制环境重新安装本包，并在 `web` 重新执行 `npm ci`、`npm run build`，再使用原来的存储与机器配置启动。检查健康端点、CPU 示例和你实际使用的原生最小配方。

备份应在服务停止后复制整个存储目录，而不只复制 SQLite 主文件；它采用 WAL 模式，还要保留运行工件。另行备份机器配置、共享配方及权重来源信息。跨机器迁移代码不会自动迁移原生环境或改写路径；现有历史数据与机器资源不要直接覆盖。

| 现象 | 检查与处理 |
|---|---|
| `/api/v1/health` 正常，但网页根路径 404 | 先构建 `web/dist`，再启动服务 |
| CLI 连接失败 | 检查服务、端口、SSH 隧道及 `WMS_BASE_URL`；地址不要重复含 `/api/v1` |
| 配方显示未配置 | 查看 `configuration_error`；填写对应机器文件或用 `runtime.config_file` 覆盖 |
| `ModuleNotFoundError` | 从运行记录确认实际 `runtime.python`；在该环境安装原生依赖，或补充正确的 `python_paths` |
| GPU/资源初始化失败 | 阅读工作进程日志，核对驱动、原生依赖、设备可见性和模型专有路径；CPU 示例通过不代表这些已配置 |
| 源实验已完成但没有评价分数 | 查看 `automatic_evaluation_id`、评价队列、指标列表和 `metric_errors` |
| `TIMEOUT` | 调整本次 `timeout_s` 或机器文件默认值；超时从工作进程启动开始计算 |
| 重启后记录为 `interrupted` | 确认原因，再显式重跑；恢复 checkpoint 需原生训练入口自行支持 |
| 文件已生成但 Web 没有列出 | 检查它是否写在当前 `output_dir` 并被列入返回的 `outputs` |

本仓库 CI 覆盖 Windows/Linux 的 CPU 测试和 Web 构建，不会替你验证目标机器的 GPU 环境、权重、长训练效果或安全研究结论。部署验收应记录具体模型、输入、运行配置、产物及评价来源。
