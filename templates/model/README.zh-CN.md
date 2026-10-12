# 新模型接入：代码模板与开发教程

适用当前 WM-SafetyLab **0.3.0** 简化接口，2026-10-11。只增加本模型的目录、普通 Python 类和 JSON 配方，不修改 `core/`、API、网页。完整平台接口见 [开发手册](../../docs/DEVELOPMENT.zh-CN.md)。

本模板有两部分：**可直接运行的 CPU 教学模型**与**需要填入官方实现的真实模型骨架**。教学模型是一维 `x_next = x + gain * action`；默认增益由人指定，训练例只拟合标量增益。它用于验证接入流程，不是神经 WM、论文 baseline、真实机器人实验，也不能据此报告安全收益。

## 1. 交付什么文件

| 文件 | 做什么 | 接入真实模型时改什么 |
|---|---|---|
| `wms_model_template/adapter.py` | 可运行的 `TeachingStateModel` | 阅读构造、预测、工件写入方式 |
| `wms_model_template/upstream.py` | `UpstreamWorldModel` 真实模型骨架 | 加载官方模型、权重，完成实际推理与格式转换 |
| `wms_model_template/train.py` | `task:native` 包装和真实执行的教学训练 | 换成官方训练函数；保存 checkpoint、数据与父模型来源 |
| `wms_model_template/metrics.py` | 独立评价，计算一维位置 MAE | 按模型任务写真正适用的指标 |
| `registry.json` | 模型与配方名称；附加模块搜索路径 | 名称加本模型前缀，指向自己的包 |
| `runtime.example.json` | 机器配置样板 | 复制成 `runtime.local.json`，填实际 Python、权重、设备 |
| `configs/predict.json`、`train.json` | 已注册配方的调用示例 | 覆盖这次任务的输入和参数 |
| `configs/predict.inline.json` | 完整配置，无需预注册名称 | 可直接交给已有服务；代码需先在该服务机器上可见 |

`__init__.py` 只标识 Python 包。整个模板没有新建执行框架，不要求继承多层类。

## 2. 先跑教学例

以下命令在 **Windows 本地**执行，另起一个教学服务；使用独立存储，避免和正式服务器实例共用数据库。端口若已使用，换一个空闲端口并同步修改客户端地址。

终端 A：

```powershell
. K:\world-model\use-k-drive.ps1
Set-Location K:\world-model\deployments\wm-safetylab
& .\.venv\Scripts\python.exe -m wms serve --port 19021 --storage-root K:/world-model/deployments/wm-safetylab/results/onboarding-manual-data --registry templates/model/registry.json
```

终端 B：

```powershell
. K:\world-model\use-k-drive.ps1
Set-Location K:\world-model\deployments\wm-safetylab
$env:WMS_BASE_URL = 'http://127.0.0.1:19021'
& .\.venv\Scripts\python.exe -m wms run templates/model/configs/predict.json --wait
```

预期：模型运行完成，预测约为 `[[0.1], [0.3], [0.2]]`，保存 `prediction.json`。模型结束后，配置里的指标会触发**另一条评价任务**。`run --wait` 只等待模型；查看 `runs` 中的评价任务，或单独手工发起：

```powershell
& .\.venv\Scripts\python.exe -m wms runs
& .\.venv\Scripts\python.exe -m wms evaluate <模型运行ID> --wait
```

后一个命令会新建一次手工评价，并不是等待原来的自动评价。评价成功时 `template_state_mae_m` 接近零。原始模型 `result.json` 不会被改写；新评价有自己的 `metrics.json` 和 `prediction_errors.json`。

### 已有服务不重启，也可以试

将本模板代码放到**服务实际所在机器**的 `templates/model/` 下，然后提交 `configs/predict.inline.json`。完整配置包含入口、模块路径和指标，不依赖新配方名称：

```powershell
& .\.venv\Scripts\python.exe -m wms run templates/model/configs/predict.inline.json --wait
```

这里的 `runtime.python_paths: ["templates/model"]` 由平台相对**服务端项目根目录**解析，不是相对客户端当前目录。远程服务看不到客户端 K 盘文件；要先同步模板代码到服务器。`runtime.python: auto` 使用服务默认工作解释器，不会自动发现 Conda/venv 或安装依赖。

注册表文件只在服务启动时读取。想让新配方出现在下拉列表中，独立开发服务通过 `--registry` 加载即可；正式服务应安排正常更新。不要在服务运行中修改 JSON 后期待热加载。

## 3. 一个模型到底要实现什么

```python
class YourModel:
    def __init__(self, config):
        # 在工作进程内导入官方库、加载权重。
        ...

    def predict(self, request):
        # request = inputs + params + seed + output_dir
        # 同名字段 params 覆盖 inputs。
        return {"predictions": future_states}

    def close(self):
        # 释放本组件自己创建的资源。
        ...
```

| 你希望运行的任务 | 实现方法 | 不应暗含的能力 |
|---|---|---|
| `predict`，未来状态／图像预测 | `predict(request)` | 不代表有决策器或控制器 |
| `generate`，视频／文本生成 | `generate(request)` | 文本到视频不自动等于动作条件 WM |
| `rollout`，直接动作策略 | `act(observation, instruction=None)` | 输出动作不证明模型内部存在 WM |
| `plan`，让已有规划器调用预测 | `predict(request)`；另外提供环境、规划器、代价函数 | 所有模型并非都能预测候选动作的后果 |
| `native`，已有训练／后训练／复杂评测 | 顶层函数 `(config, output_dir, context)` | 平台包装不自动实现论文训练算法 |

只实现实际需要的方法。需要潜在记忆的模型，可在实例中保存并明确初始化/更新规则；不要把任意视频 latent 直接解释成机械臂状态。当前 `plan` 为每组候选调用 `predict`，带内部记忆的模型应避免候选之间相互污染，通常需从同一真实观测编码出的状态复制后推演。

### 形状和单位是开发者共同约定

教学模型严格约定：观测 `[position_m]`；动作序列 `[H, 1]`，元素为位移增量（m）；输出 `predictions` 为 `[H, 1]`，第 `t` 行对应**执行第 `t` 个动作之后**的状态，不包含初始状态。

真实接入需在自己的 README 写清：观测字段、坐标系、动作绝对值/增量、单位、控制频率、归一化、预测时域、张量维度以及模型是否一次只支持一个环境。平台当前不替你验证这些语义。比如 Gymnasium Pendulum 的三维角度编码和力矩动作，不能直接与本模板一维位置模型组合；能导入两个类不代表可配对实验。

## 4. 替换为真实官方模型

1. 将官方仓库和原生环境放在 `K:/world-model/repos/`、`K:/world-model/envs/` 下；保持上游源码干净。先用官方方式独立跑一个最小输入。
2. 复制 `upstream.py` 到自己的包名，例如 `my_dreamer/adapter.py`。在 `__init__` 导入实际后端并加载权重，不要在服务/API 模块顶层加载 Torch/JAX 或 GPU。
3. 完成观测、动作和输出转换。保留上游原始输出文件，并另存平台展示需要的文件。
4. 复制 `runtime.example.json` 为机器专属配置，填写实际路径，修改自己的配方名和入口。
5. 先做单次 `predict`，确认与官方直接调用一致，再加独立指标，最后考虑环境组合和规划。

机器配置示意，必须替换所有占位项：

```json
{
  "python": "K:/world-model/envs/MY_MODEL/Scripts/python.exe",
  "checkpoint": "K:/world-model/weights/MY_MODEL/ACTUAL_CHECKPOINT",
  "upstream_revision": "ACTUAL_COMMIT",
  "device": "cuda:0",
  "env": {"CUDA_VISIBLE_DEVICES": "0"},
  "timeout_s": 600
}
```

Linux 对应改成 `/home/.../world-model/envs/MY_MODEL/bin/python` 及服务器本地权重路径。不要将 Windows 机器配置覆盖服务器配置。机器配置中的相对权重路径，模板按**机器配置文件所在目录**解析；真实权重既可为文件，也可为官方目录形式。

`Registry` 自动处理的是解释器、环境变量、cwd、python_paths 等标准运行字段，**不会自动把 checkpoint/device 传入模型**。模板通过 `model.config.runtime_config: "$runtime.config_file"` 把机器文件路径交给适配器读取。真实适配器加载后应记录最终权重路径、上游 revision、运行参数和输出语义。

当前 `onboarding-model-upstream` 默认没有 `runtime.local.json`，因此显示未配置。即使填入了真实路径，骨架仍会 `NotImplementedError`，直到你实现真实推理；不会用合成结果冒充接入成功。

PyTorch 适配的典型实现顺序如下；函数名和张量格式须替换为实际官方 API，不是可以直接运行的上游实现：

```python
def __init__(self, config):
    import torch
    self.torch = torch
    self.backend = load_official_checkpoint(...)  # TODO：真实 API
    self.backend.eval()

def predict(self, request):
    observation, actions = convert_to_backend_tensors(...)  # TODO：语义转换
    with self.torch.inference_mode():  # 或 torch.no_grad()
        output = self.backend.predict(...)  # TODO：真实 API
    return {"predictions": convert_prediction(output)}
```

若是 JAX、远程推理或扩散后端，采用其原生推理方式，不照搬 PyTorch 的 `eval()`。固定 seed 也不能保证所有 GPU 算子位级确定性。

## 5. 训练、加载新权重、独立评价

先运行已实现的教学训练：

```powershell
& .\.venv\Scripts\python.exe -m wms run templates/model/configs/train.json --wait
```

得到 `checkpoint.json`（拟合增益约 0.8）、`training_data.json`（64 条真实用于拟合的合成数据）、`training.json`（后端、参数、随机种子、解释器、数据来源）。这验证训练包装、工件保存与权重回载，**没有留出集，因此不能用训练例误差当泛化结论**。

下一次预测使用完整模型配置并填写**训练运行机器上的 checkpoint 实际绝对路径**：

```python
import wms

record = wms.run({
    "recipe": "onboarding-model-predict",
    "model": {
        "entrypoint": "wms_model_template.adapter:TeachingStateModel",
        "config": {"checkpoint": "REPLACE_WITH_REAL_SERVER_CHECKPOINT_PATH"},
    },
    "inputs": {"reference": [[0.08], [0.24], [0.16]]},
})
print(wms.wait(record["id"]))
```

替换参考值是因为合成训练数据明确使用增益 0.8，不应将不同数据生成规则的“真值”混用。实际研究需采用冻结的独立测试集比较训练前后模型，不能根据模型输出修改真实参考标签。

真实训练接入保持 `train.py:run` 的薄包装思路，在自己的包实现：

```python
def train_backend(inputs, params, output_dir, context):
    # 读取真实数据、父checkpoint，调用官方训练，定期检查取消。
    # 训练产物写入output_dir；不覆盖父checkpoint。
    ...
    return {"checkpoint": str(saved_checkpoint_file), "dataset": data_version}
```

通过 `params.train_entrypoint` 调用它。如果上游只提供命令行脚本，可在这个函数内用明确参数列表启动；持续等待子进程、检查返回码，不要后台分离后提前报完成。当前教学包装接受一个 checkpoint 文件；多文件权重目录要在自有包装中列出必要文件或生成一个权重索引文件，不能把目录假装成单个文件工件。

`parent_checkpoint`、`upstream_revision` 在模板中仅作来源记录，不会自动加载或执行后训练。真实后端必须显式消费父权重、数据和训练参数。平台不代替 train/eval 划分、学习率设置、训练算法或论文验收。

## 6. 结果与指标约定

小型预测可返回 JSON `{"predictions": ...}`。为了兼容 `plan`，不要只把预测放在 `{"data": ...}` 里面；规划器读取顶层 `predictions`。大型张量、潜在状态、视频保存为本次 `output_dir` 的文件，返回文件引用：

```python
return {
    "outputs": [{"kind": "video", "path": "video.mp4", "media_type": "video/mp4"}],
    "data": {"frames": 49, "fps": 16, "checkpoint": checkpoint_id},
}
```

工件必须在当前运行目录内且真实存在。视频用 mp4，数组可用 npz，结构数据用 JSON，轨迹应记录时间/动作/状态语义；不要把大数组塞入 API 结果。

`prediction_metrics(result, params)` 收到两个不同目录：

- `source_output_dir`：模型原始输出目录，只读取。
- `output_dir`：本次独立评价目录，只在此写指标报告或图。

这是开发约定，当前系统没有操作系统级只读隔离。缺少参考真值应报告错误或明确不可评价，不填零分。不要在推理适配器内混入昂贵评价；指标由独立任务执行，可换参数重评，不覆盖原运行。当前模板输出 `template_state_mae_m`，真实模型要设计适用指标，视频质量、物理预测和闭环安全指标不能混为一个分数。

教学 `save_predictions` 默认关闭，单次预测配方打开它。规划时会多次调用预测，不应每个候选覆盖同一个预测文件；规划轨迹以 `plans.json`/`trajectory.json` 为准，若要存所有候选应使用独立编号并控制大小。

## 7. 交付前替换清单

- [ ] 独立包名、配方名无冲突；`python_paths` 指向实际工作进程可见目录。
- [ ] 官方最小命令先跑通；记录上游 commit、依赖版本、权重和数据来源。
- [ ] 真实推理替换全部 TODO/NotImplemented；没有用教学输出替代失败的真实后端。
- [ ] 观测／动作／预测的形状、单位、归一化和时间对齐写清。
- [ ] 随机种子、设备、checkpoint 使用结果可追溯；CPU 演示与 GPU 实跑分别记录。
- [ ] 原始结果和独立评价分离；缺失输出或真值不默认为成功。
- [ ] 若配仿真器，完成同任务同动作下的预测／真实后果对齐。
- [ ] 模型与环境依赖能安装在同一解释器中；否则另做明确的跨进程模型服务适配，当前模板不会自动解决。

平台已有真实 ManiSkill 状态 WM 例位于 `src/wms/engines/state_wm_maniskill/`。那是特定任务的工程实验；本模板更适合成员先学习公共接入接口，再按自己的原生模型补真实逻辑。
