# WM SafetyLab 当前开发手册

TD-MPC2 实际接入（2026-10-12）：[PushCube 中文开发教程](TD_MPC2_PUSHCUBE_GUIDE.zh-CN.md) 与 [正式验收报告](TD_MPC2_PUSHCUBE_ACCEPTANCE_20261012.zh-CN.md)，覆盖真实训练、checkpoint 回载、三组闭环评测、录像、潜空间对照和独立指标；适用范围为单训练种子的工程基线。

新增接入教程（2026-10-11）：[模型与仿真环境开发总指南](ONBOARDING_MODELS_AND_ENVIRONMENTS.zh-CN.md)，内含两套可复制代码模板和已实跑的端到端示例。

更新：2026-10-12，代码版本 0.3.0。本文是现行接口与协作依据。部署从[部署与使用指南](DEPLOYMENT_AND_USAGE.zh-CN.md)开始；各模型实际验证范围见[模型与实验手册](MODELS_AND_EXPERIMENTS.zh-CN.md)。旧设计及完整原始实验记录保留在维护者本地，不随公开代码发布。代码已实现与真实模型已验收是不同状态。

## 1. 开发目标和边界

统一提交、运行、输出和可选评价，尽量复用已有原生代码。新增模型的主要工作是写一个普通 Python 组件或原生函数，再提供 JSON 配方。

运行链如下：

```text
Web / CLI / Python
    → HTTP API
    → Registry.resolve：按名称展开本次配置
    → Runner：排队，启动 runtime.python
    → UnifiedWorker：执行本次任务
    → result.json + 日志 + 原始输出
    → 可选 evaluate 任务加入同一队尾
    → 独立评价工作进程 → 独立结果、指标和错误
    → SQLite / HTTP / Web 对照显示
```

当前为单服务、单机 FIFO 队列，同一时刻执行一个模型或评价工作进程。模型和指标依赖均在工作进程中导入，不在 Web 服务进程中计算指标。同一存储目录不能启动多个服务实例或多个 Uvicorn worker。不同任务可以选择不同 Python；同一次模型／环境／规划器组合仍要在一个可共同导入这些依赖的环境中执行。

当前不默认提供：全组合兼容矩阵、入口白名单、规范桥接回放、四组角色约束、证据哈希封存、通用防护权限系统、分布式 GPU 调度。仍保留运行实际需要的错误记录、停止、环境关闭、输出路径定位。

## 2. 代码职责

| 文件或目录 | 职责 | 何时修改 |
|---|---|---|
| `src/wms/__init__.py` | 小型 Python HTTP 客户端 | 增加公共客户端操作 |
| `src/wms/__main__.py` | `python -m wms` 入口 | 通常无需修改 |
| `app/api.py` | 服务组装、HTTP 路由、Web 静态文件 | 增加用户操作或查询 |
| `app/cli.py` | 与 HTTP 对应的命令 | 增加 CLI 命令 |
| `core/registry.py` | 读取注册表、展开配方和组件、选择 Python | 改配置查找逻辑 |
| `core/protocols.py` | 模型、环境、规划器的结构约定 | 改实际调用签名 |
| `core/runner.py` | 模型／评价共用队列、超时、取消和状态记录 | 改运行管理 |
| `core/processes.py` | 进程身份核对与所属进程树清理 | 改异常恢复 |
| `core/storage.py` | 每次运行一条 SQLite 记录和一个目录 | 改存储 |
| `core/evaluation.py` | 构建评价任务、子进程执行指标、带来源的比较 | 改通用评价入口 |
| `core/planners.py` | 可选 Random Shooting 与 CEM | 改已有规划算法 |
| `core/payloads.py` | 输出值与原生载荷兼容辅助 | 处理现有后端输出 |
| `runners/unified.py` | 6 种任务的组件调用、握手和结果归一化 | 增加真正不同的执行方式 |
| `engines/native_tasks.py` | 通用原生预测包装与旧入口兼容导出 | 通常无需为新模型修改 |
| `engines/<模型>/native.py` | 本模型原生任务的薄包装入口 | 接入本模型代码 |
| `engines/<模型>/` | 模型实际调用、状态/动作转换 | 模型方向开发 |
| `scenarios/maniskill/` | ManiSkill 环境和固定策略适配 | 环境方向开发 |
| `web/src/` | 实验、运行、结果、对比四页 | 改交互与展示 |
| `examples/minimal/` | 可直接阅读和运行的 CPU 合成组件 | 理解接口、添加简短示例 |
| `native/<模型>/recipes.json` | 独立、可共享的模型配方 | 新增或修改本模型默认任务 |
| `native/<模型>/runtime*.json` | 机器专属解释器、权重和资源位置 | 在目标机器填写，不跨机器覆盖 |

研究上的四层通过职责和标签表达：

| 研究层 | 综述阶段 | 开发落点 |
|---|---|---|
| 数据层 | S1 数据与观测 | 数据文件、环境 reset、inputs |
| 模型层 | S2 状态与记忆；S3 动力学与想象 | 模型组件内部状态和 predict / generate |
| 算法层 | S4 评估与规划 | 规划器、代价函数和方法函数 |
| 执行与反馈层 | S5 动作与控制；S6 反馈与更新 | act / step、轨迹、评价、原生更新任务 |

例如 `tags: ["S3", "S4", "non_adversarial_failure"]` 是研究标注，不会触发权限或兼容门禁。防护可先作为组件内部函数、包装组件或独立原生配方；不要求继承统一防护基类。

## 3. 一份实验配置

最短调用可以只给配方名和覆盖项：

```json
{
  "recipe": "demo-rollout",
  "name": "交互运行 seed 7",
  "seed": 7,
  "steps": 12,
  "metrics": []
}
```

注册表会展开为完整配置。也可以直接提交：

```json
{
  "name": "单次预测",
  "task": "predict",
  "direction": "control",
  "runtime": {
    "python": "auto",
    "python_paths": [],
    "env": {}
  },
  "model": {
    "entrypoint": "my_project.model:MyModel",
    "config": {"checkpoint": "/path/to/checkpoint"}
  },
  "inputs": {"observation": [0], "actions": [[0.1], [0.2]]},
  "params": {},
  "seed": 0,
  "steps": 2,
  "metrics": [],
  "tags": ["S3"]
}
```

`/path/to/checkpoint` 是示意路径，需换成实际资源。Windows 项目资源使用 K 盘路径，JSON 中可用正斜杠。

| 字段 | 用途 |
|---|---|
| `task` | 必填：generate / predict / rollout / plan / native；独立评价使用 evaluate |
| `name`、`direction` | 显示名称和研究方向 |
| `runtime.python` | 实际工作进程解释器；省略或 auto 使用服务指定的默认解释器 |
| `runtime.config_file` | 机器配置 JSON 的引用；本次请求可以覆盖配方默认值 |
| `runtime.native_key` | 查找启动参数 --native-configs 中对应项的键 |
| `runtime.python_paths` | 工作进程需要的额外模块搜索路径，推荐绝对路径 |
| `runtime.cwd` | 可选工作目录；默认本次运行目录 |
| `runtime.env` | 子进程环境变量，例如 CUDA_VISIBLE_DEVICES |
| `model/environment/algorithm` | 注册表名称，或含 entrypoint 和 config 的对象 |
| `entrypoint` | native 任务的函数入口 |
| `inputs` | 本次输入、观测、动作序列、指令等 |
| `params` | 本次算法或原生任务参数 |
| `seed` | 传给模型请求或环境 reset；组件负责自己的随机数初始化 |
| `steps` | rollout / plan 的最大交互步数；native 函数按自身语义读取 |
| `timeout_s` | 可选执行时限，正数秒；从工作进程启动计时，不计排队；优先于 runtime.timeout_s |
| `metrics` | 指标入口清单；模型完成后自动创建评价，空列表不自动评价 |
| `tags` | 研究标签，用于表达 S1–S6 或实验类别 |

`predict` 的预测长度由实际 `inputs.actions`、horizon 或原生模型请求定义，并非所有模型都会读取顶层 `steps`。原生函数的 inputs / params 由该函数说明。

## 4. 六种任务

| task | 调用 | 必需组件 | 输出 |
|---|---|---|---|
| generate | `model.generate(request)` | model | 视频、图片、文本或其他文件 |
| predict | `model.predict(request)` | model | 任意 JSON 预测或输出文件；environment 可选 |
| rollout | `env.reset → model.act → env.step` | model、environment | trajectory.json |
| plan | `预测候选动作 → cost → planner.plan → env.step` | model、environment、algorithm | trajectory.json、plans.json |
| native | `function(config, output_dir, context)` | 顶层 entrypoint | 原生实验、训练、后训练的输出 |
| evaluate | `measure(source_result, params)` | 来源运行和 metrics | 独立 metrics.json、result.json 及评价产物 |

离线预测不要求仿真环境。predict 配置了 environment 时，需提供 `inputs.actions`：worker 从环境 reset 获取初始观测，把同一组动作交给模型和环境，保留预测与实际转移。

generate / predict 的 request 由 `inputs` 与 `params` 合并，再加入 `seed` 和 `output_dir`；重名时 params 覆盖 inputs。rollout 使用 `inputs.instruction` 作为 act 的可选指令。

通常通过 `wms.evaluate(source_id)` 或 `POST /runs/{id}/evaluate` 创建评价，让服务填写 source_run 和来源目录，不必手写 evaluate 配方。

## 5. 新增模型组件

模型、环境、规划器使用普通 `Class(config)` 构造。worker 会在 config 中加入本次 `output_dir`。不需要继承平台基类，只实现所选任务调用的方法。

```python
from pathlib import Path

class MyModel:
    def __init__(self, config):
        self.config = config
        # 在这里导入模型库并加载权重。

    def predict(self, request):
        observation = request["observation"]
        actions = request["actions"]
        predictions = self.predict_actions(observation, actions)
        return {"predictions": predictions}

    def generate(self, request):
        output = Path(request["output_dir"]) / "video.mp4"
        # 调用模型，将视频写入 output。
        return {"outputs": [{"kind": "video", "path": str(output)}]}

    def act(self, observation, instruction=None):
        # 返回环境控制器实际接受的动作。
        return self.policy(observation, instruction)

    def close(self):
        # 释放该组件拥有的资源。
        pass
```

上面是签名示意，`predict_actions`、`policy` 和视频生成需由接入者实现。不会为每个模型强制要求 generate / predict / act 全部存在。

模型输出可以是语言状态、视频、图像、潜变量或动作。统一的是外层调用和文件引用，不假定不同模型具有相同的内部世界状态。VLA 的 act 也不会被自动视为世界模型预测。

## 6. 新增环境组件

```python
class MyEnv:
    def __init__(self, config):
        self.config = config
        # 初始化实际仿真环境。

    def reset(self, seed, inputs):
        # 配置本次场景，设置随机种子，返回观测本身。
        return observation

    def step(self, action):
        return {
            "obs": next_observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "cost": cost,
            "info": info,
        }

    def close(self):
        pass
```

这是平台的字典转移格式。Gymnasium / ManiSkill 等原始 reset/step 元组需要在环境适配器中转换。cost、info 以及其他研究字段可选；obs 是继续交互所需字段。所有写入结果的值需可转为 JSON，较大的张量或视频保存为文件。

动作单位、控制模式、归一化和形状转换应放在对应适配器中。取消全局规格框架不代表这些转换可以省略。

## 7. 可选规划器和方法

规划器构造为 `Planner(config)`，实现：

```python
def plan(self, observation, step, context):
    # context.predict(observation, action_sequence) -> predictions
    # context.cost(observation, predictions, inputs) -> scalar
    # context.action_range -> 每个动作维度的 [下限, 上限]
    return first_action
```

内置 `wms.core.planners:RandomShooting` 和 `wms.core.planners:CrossEntropy`，低代价候选优先。plan 配方中填写：

```json
{
  "algorithm": {
    "entrypoint": "wms.core.planners:RandomShooting",
    "config": {"horizon": 3, "candidates": 8}
  },
  "params": {
    "cost": "my_project.metrics:planning_cost",
    "action_range": [[-0.2, 0.2]]
  }
}
```

代价函数签名为 `planning_cost(observation, predictions, inputs) -> float`。合成参考在 `examples/minimal/wms_demo.py`。预测与动作编码必须由模型和场景开发者明确配合；平台不尝试自动推断。

## 8. 包装原生论文流程

不适合拆成通用组件的训练、后训练、整篇论文流程，优先使用 native：

```python
def run_paper(config, output_dir, context):
    context.check_cancelled()
    context.event("progress", {"phase": "prepare"})
    # 调用现有代码，把结果放在 output_dir 中。
    report = {"message": config.get("inputs", {}).get("message", "")}
    context.write_json("report.json", report)
    return {
        "outputs": [{"kind": "report", "path": "report.json"}],
        "data": {"finished": True},
    }
```

对应配置：

```json
{
  "name": "原生论文实验",
  "task": "native",
  "entrypoint": "my_project.task:run_paper",
  "runtime": {"python": "/path/to/native-venv/bin/python"},
  "inputs": {},
  "params": {}
}
```

`context` 提供 output、check_cancelled、event、write_json。长循环中主动检查取消；不要在模块导入时直接开始训练。尽量保留原生项目代码，适配放在本平台或独立研究模块中。

原生入口已拆到 `engines/wan/native.py`、`engines/maniskill/native.py` 等各自目录，默认参数放在 `native/<模型>/recipes.json`。`engines/native_tasks.py` 保留旧入口导出及通用预测包装；新模型不需要加入共享模型列表。通用原生预测器通过配置中的 `adapter_entrypoint` 加载模型适配类，不再依赖固定 ADAPTERS 表。

模型负责人交付自己的入口、配方和配置示例即可。服务器现有 `runtime.json`、`runtime.linux.json`、虚拟环境、权重、数据和历史结果应原位保留。新增配方不等于该机器已经安装模型依赖。

## 9. 注册配方与机器配置

服务启动时依次加载 `examples/minimal/registry.json`、按路径排序的 `native/*/recipes.json`，再加载显式的 `--registry`／`--components` 文件。两个参数是同一选项的别名，均可重复指定。

- 同一实际文件只加载一次；同组重复 ID 报错并给出两个来源，不能靠加载顺序覆盖。
- 支持 `models/environments/algorithms/runtimes/recipes` 五组；可以只填写需要的组。
- 配方在服务启动时加载。修改共享默认配方需重启；每次请求覆盖 inputs、params、seed、steps 等无需重启。
- 文件中的相对 runtime.config_file、cwd、python 路径及 python_paths 以声明文件为基准；HTTP 内联请求中的相对运行路径以平台项目根目录为基准。推荐机器资源使用绝对路径。
- 顶层 python_paths 提供该文件的组件导入路径，仅用于工作进程，不再注入服务的 sys.path。

一个新增模型的目录可以是：

```text
native/my_model/
  recipes.json           团队共享
  runtime.example.json   团队共享的填写示例
  runtime.json           本机实际值，不覆盖其他机器的文件
src/wms/engines/my_model/
  native.py              适配自己的模型流程
```

`recipes.json` 示例：

```json
{
  "recipes": {
    "my-model-predict": {
      "name": "我的模型预测",
      "task": "native",
      "direction": "control",
      "entrypoint": "wms.engines.my_model.native:predict",
      "runtime": {
        "native_key": "my_model",
        "config_file": "runtime.json"
      },
      "inputs": {},
      "params": {},
      "metrics": [],
      "timeout_s": 600
    }
  }
}
```

机器配置示例（路径需替换成该机器实际值）：

```json
{
  "python": "/home/team/world-model/envs/my_model/bin/python",
  "env": {"CUDA_VISIBLE_DEVICES": "0"},
  "checkpoint": "/home/team/world-model/weights/my_model"
}
```

Registry 只提取 python、env、cwd、python_paths、timeout_s 作为运行设置；checkpoint 等原生字段仍由模型入口读取所选文件。入口从 `config["runtime"]["config_file"]` 获取实际文件，避免解释器和权重配置引用不同文件。组件构造配置中可用 `"$runtime.config_file"` 和 `"$runtime.env"` 引用已经解析的值；已有 ManiSkill 配方提供了完整例子。

服务器可以保留配置映射：

```json
{
  "my_model": "/home/team/world-model/deployments/wm-safetylab/native/my_model/runtime.linux.json",
  "maniskill": "/home/team/world-model/deployments/wm-safetylab/native/maniskill/runtime.linux.json"
}
```

```powershell
. K:\world-model\use-k-drive.ps1
Set-Location K:\world-model\deployments\wm-safetylab
.\.venv\Scripts\python.exe -m wms serve --registry K:\world-model\my-research\recipes.json --native-configs K:\world-model\machine-configs.json
```

选择机器文件的顺序：

1. 请求明确给出的 runtime.config_file。
2. 兼容现有调用的请求 params.profile 或 params.runtime_config。
3. `--native-configs` 中与 runtime.native_key 对应的文件。
4. 配方 runtime.config_file 默认文件。

文件选定后，运行设置按“机器文件默认值 → 配方显式 runtime → 请求显式 runtime”合并，后者优先；env 按键合并。原生文件的 cuda_visible_devices／gpu_device 可以补充 CUDA_VISIBLE_DEVICES，显式 env 优先。`--native-configs` 文件内的相对路径以映射文件位置为基准。

旧 world_model 集合只在显式传入 `--native-configs` 的 world_model 映射时用于兼容选择配置，不再用于默认发现模型配方；历史文件保留。

GET /recipes 与 GET /recipes/{id} 返回 configured 和 configuration_error。机器文件缺失时配方仍可查看、编辑；补充有效运行配置后可提交。configured=true 仅表示配置可读取，不能当作依赖、模型或实验已验证。

## 10. 原始结果与独立评价

worker 允许返回普通 JSON 值，自动放入 data；也允许显式返回：

```json
{
  "status": "completed",
  "outputs": [
    {"kind": "video", "path": "video.mp4", "media_type": "video/mp4"},
    {"kind": "trajectory", "path": "trajectory.json", "media_type": "application/json"}
  ],
  "metrics": {"training_loss": 0.12},
  "data": {"steps": 8}
}
```

outputs 中的文件必须真实存在于本次 output_dir 下。相对路径以 output_dir 为起点；合法绝对路径会转成相对路径。大张量保存为 npy／npz 等文件再返回引用。模型原生 metrics 保留在原始结果中，后续评分不与它们混合。

一次独立评价也是一条运行：task=evaluate，source_run 指向来源实验。自动评价仅在模型正常完成且 metrics 非空时产生，加入同一 FIFO 队尾；模型完成记录与 automatic_evaluation_id 在同一数据库事务中保存，避免重启重复触发。手动重评分可以产生多次评价。

评价默认继承来源的 Python 和运行设置，也可以指定独立 runtime 和 timeout_s。开始评价前模型进程已经退出；继承解释器不表示模型继续占用显存。服务不导入、执行指标。

指标函数保持 `measure(result, params)`，两条路径必须区分：

```python
import json
from pathlib import Path

def trajectory_length(result, params):
    # 读取来源模型工件。
    source = Path(result["source_output_dir"]) / "trajectory.json"
    trajectory = json.loads(source.read_text(encoding="utf-8"))
    values = {"steps": len(trajectory["steps"])}
    # 本次评价的新增文件只写入本次评价目录。
    report = Path(result["output_dir"]) / "length_report.json"
    report.write_text(json.dumps(values), encoding="utf-8")
    return values
```

source_output_dir 是来源实验目录，只读；output_dir 是本次评价输出目录。旧指标若从 result["output_dir"] 读取模型文件，必须迁移为 source_output_dir。框架不再改写来源 result.json；这是受信任研究代码的写入约定，并非操作系统级只读沙箱。

指标配置示例：

```json
[
  {
    "name": "trajectory_length",
    "entrypoint": "my_project.metrics:trajectory_length",
    "params": {},
    "unit": "steps",
    "version": "1"
  }
]
```

也支持入口字符串。返回字典时生成多个指标，标量按配置名称记录；不同函数返回同名指标会记为错误，不覆盖已有值。建议记录 unit 和 version，便于团队对比。

普通指标异常记录在评价 result.metric_errors，其他指标继续计算。评价进程正常完成但有指标错误时，state 仍为 completed，evaluation_status 为 partial_failed；Web 标明“部分指标失败”。进程崩溃、超时或取消使用该评价自己的 failed／cancelled 状态。上述状态都不会倒改来源模型。

比较默认选择来源实验最近一条 completed 评价，包括有部分错误的评价。错误项保持缺失，不从旧评价补值。若还有较新的 queued、running、failed 或 cancelled 尝试，响应同时给出 latest_evaluation 和提示。显式评价 ID 必须属于该来源且已完成；未完成记录可在评价历史或运行页查看。

新记录带 record_version=3。缺少该字段或小于 3 的旧记录，其 result.metrics 只读显示为 legacy；新记录的原生指标标为 native。已有评价尝试但没有完成评分时，不静默回退到旧值；用户仍可显式选择合法的 legacy／native 来源。不同指标入口、参数、单位、版本或未知版本会提示，不自动排名、平均或合成统一安全分数。

## 11. HTTP / Python / CLI

默认服务地址 `http://127.0.0.1:8765`，Python 与 CLI 用 WMS_BASE_URL 覆盖。Python 客户端不会启动服务。自动评价和手动评价的等待都以评价 ID 为准：

```python
import wms

run = wms.run({"recipe": "demo-rollout", "seed": 0, "timeout_s": 60})
model = wms.wait(run["id"])  # 只等待模型，不隐式等待后续评价。
if model.get("automatic_evaluation_id"):
    automatic = wms.wait(model["automatic_evaluation_id"])

# 明确创建新的一次评分，而不是修改来源结果。
evaluation = wms.evaluate(run["id"], timeout_s=60)
scored = wms.wait(evaluation["id"])
print(scored["state"], scored.get("evaluation_status"))
print(wms.evaluations(run["id"]))
print(wms.compare([run["id"]], evaluations={run["id"]: evaluation["id"]}))
```

公共函数签名：

```python
run(config)
get_run(run_id)
wait(run_id, timeout=3600)
cancel(run_id)
retry(run_id)
evaluate(run_id, metrics=None, *, runtime=None, timeout_s=None)
evaluations(run_id)
compare(run_ids, *, evaluations=None)
```

`wait` 的 timeout 是客户端等待上限，不会取消后台任务；执行上限由任务 timeout_s 控制。默认评价使用来源 metrics；提供 metrics 可以换成新的指标清单。retry 总是复制配置创建新 ID，不覆盖原任务。

CLI 示例：

```powershell
. K:\world-model\use-k-drive.ps1
Set-Location K:\world-model\deployments\wm-safetylab
$env:WMS_BASE_URL = 'http://127.0.0.1:8765'
.\.venv\Scripts\python.exe -m wms recipes
.\.venv\Scripts\python.exe -m wms run demo-rollout --wait
.\.venv\Scripts\python.exe -m wms runs
.\.venv\Scripts\python.exe -m wms get-run RUN_ID
.\.venv\Scripts\python.exe -m wms logs RUN_ID
.\.venv\Scripts\python.exe -m wms evaluate RUN_ID --wait --timeout-s 60
.\.venv\Scripts\python.exe -m wms evaluate RUN_ID --metrics K:\world-model\metrics.json --runtime K:\world-model\evaluation-runtime.json --wait
.\.venv\Scripts\python.exe -m wms compare RUN_ID_1 RUN_ID_2 --evaluations K:\world-model\evaluation-selection.json
```

RUN_ID 等替换为实际 ID；metrics.json 是指标数组，evaluation-runtime.json 是 runtime 对象，evaluation-selection.json 是 `{"来源ID":"评价ID"}` 映射。`evaluate --wait` 遇到失败、取消、中断或部分指标失败返回非零退出码。

HTTP 路由统一前缀 `/api/v1`：

| 方法与路径 | 用途 |
|---|---|
| GET /health | 服务状态与代码版本 |
| GET /recipes；GET /recipes/{id} | 配方目录／展开配置和机器配置状态 |
| GET /components | 组件定义 |
| POST /runs | 提交配置，立即返回运行记录 |
| GET /runs；GET /runs/{id} | 全部任务／指定任务记录 |
| GET /runs/{id}/logs?offset=0 | 日志文本和 next_offset |
| GET /runs/{id}/result | 该任务自己的结果或 null |
| GET /runs/{id}/files；GET /runs/{id}/files/{path} | 工件目录／下载 |
| POST /runs/{id}/cancel；POST /runs/{id}/retry | 取消／创建新重试 |
| POST /runs/{id}/evaluate | 异步创建评价；可传 metrics、runtime、timeout_s |
| GET /runs/{id}/evaluations | 来源实验的完整评价记录，最新在前 |
| POST /comparisons | 传入 run_ids 和可选 evaluations 映射 |

评价 POST 的响应是新任务记录，不是最终指标。比较响应含 rows 和 warnings；各行含 metrics、metric_errors、evaluation（实际评分来源）及 latest_evaluation（最新尝试）。API 不把正在运行的评价伪装为已有分数。

Web 保持四页：实验编辑和提交；运行按实验／评价／全部筛选；结果分别查看原始输出与评价历史；比较选择评价记录，展示来源、时间、配置及警告。评价的日志、取消和重试直接复用运行页。

## 12. 生命周期、存储与异常恢复

任务状态为 queued、running、cancelling、completed、failed、cancelled、interrupted。completed 仅表示执行完成，不代表模型质量、任务成功率或安全结论。模型完成需正常退出、合法结果及声明工件存在；不能只根据 result.json 存在判断成功。

- timeout_s 从子进程启动开始计时，不计排队。超时终止所属进程树，记录 state=failed、error_code=TIMEOUT 和原因，继续后续任务。
- 人工取消先发出 cancel.flag；组件可以调用 context.check_cancelled 主动结束，超出宽限期由运行器终止所属进程。
- 服务记录 PID、进程创建时间、运行命令和目录。重启时只清理核对为本任务的进程，不按进程名或旧 PID 盲目停止其他程序。
- 工作进程在 start.flag 前等待服务保存身份；任务返回后通过 finish.ready／finish.flag 握手，服务记录仍存活的子进程后才允许退出。这些文件由平台管理。
- 异常重启后未完成任务标为 interrupted，包括评价；不会自动恢复训练或重新投递。需要用户显式 retry。
- 原生入口应管理自己启动的后台程序，不主动将任务进程脱离平台进程树。此机制面向平台所属科研任务，不是任意外部守护进程管理器。

```text
<storage_root>/
├── runs.sqlite3
└── runs/
    ├── <model_run_id>/
    │   ├── config.json / run.log / 握手文件
    │   └── output/
    │       ├── result.json             固定来源原始结果
    │       └── 视频、轨迹等原始工件
    └── <evaluation_id>/
        ├── config.json                 source_run、metrics、runtime
        ├── run.log
        └── output/
            ├── result.json             本次评价结果
            ├── metrics.json            值、错误及指标说明
            └── 本次评价新增文件
```

SQLite 保存任务汇总；来源结果不被独立评价改写。默认 Windows 存储是 data/，Linux 启动脚本使用 data-linux/。旧 runs.sqlite3 中的历史记录直接读取；更早 records.sqlite3／artifacts／jobs 原位保留，本轮不导入。

## 13. 测试、部署与共同开发

本机执行和安装前先加载 K 盘设置，测试临时文件也放在项目内：

```powershell
. K:\world-model\use-k-drive.ps1
Set-Location K:\world-model\deployments\wm-safetylab
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m pytest tests --basetemp results\manual-pytest-tmp
Set-Location K:\world-model\deployments\wm-safetylab\web
npm run build
```

平台环境需安装 pyproject.toml 声明的依赖，新增的 psutil 只在服务环境使用；不要求向各模型原生环境安装它。`--no-deps` 适用于依赖已经准备好的环境。

`.github/workflows/ci.yml` 定义 Windows／Linux Python 3.11 CPU 测试和 Web 构建，不自动下载 GPU 权重。本地目录目前没有 Git 元数据，CI 文件已编写不等于托管 CI 已触发；推送到团队仓库后才能检查实际运行结果。

| 负责人 | 维护范围 | 接入交付 |
|---|---|---|
| 平台核心 | Registry、Runner、Store、API、公共客户端 | 配置规则、队列和状态语义、核心测试 |
| 模型方向 | 自己的 engines/<模型>、native/<模型> 配方 | 原生可运行入口、依赖和数据说明、一个小规模实例 |
| 环境方向 | scenarios/<环境>、控制器和轨迹转换 | reset/step/close、状态／动作单位、环境调用实例 |
| 集成负责人 | Web、评价、对照实验与回归记录 | 来源明确的指标、可复现命令、成功和失败证据 |

2–3 人可合并角色，但同一阶段不要多人同时修改 Runner／Store。接入顺序是“原生跑通 → 薄包装 → 独立配方 → 平台运行 → 独立评分”。服务核心无需随着每个模型接入增加分支。

服务器升级只同步平台代码、共享配方和前端构建；先备份源码并检查无在途任务，再切换单一服务。保留服务器机器配置、环境、权重与历史结果，不将 Windows 配置复制覆盖到 Linux。每次新版本分别记录 CPU、服务器原生、浏览器和研究实验的证据，不能沿用旧版通过记录宣称新版全部通过。
