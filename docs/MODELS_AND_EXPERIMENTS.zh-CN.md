# 模型、仿真环境与实验使用手册

适用版本：WM-SafetyLab 0.3.0；代码与既有运行记录核对日期：2026-10-12。

本文覆盖仓库当前提供的所有默认配方、可选教学配方，以及各模型从资源准备到运行、读取输出和解释结果的方法。平台安装、前端和服务启动先看 [README](../README.md)；开发新组件看 [接入指南](ONBOARDING_MODELS_AND_ENVIRONMENTS.zh-CN.md) 和 [开发文档](DEVELOPMENT.zh-CN.md)。

**仓库发布的是代码和配置模板，不包含模型权重、数据集、机器环境或历史运行工件。** 文中的实测结果是已有部署的观察，不代表刚克隆仓库就能执行全部配方，也不代表本次文档更新重新跑过全部模型。

## 1. 先选择需要的能力

默认注册表会加载 `examples/minimal/registry.json` 和 `native/*/recipes.json`，共 19 条配方。配方名称不决定执行器类型：例如三个 `predict-*` 兼容适配器仍然使用 `task=native`。

| 模型或环境 | 当前配方 ID | 实际能力 | 已有验证范围 |
|---|---|---|---|
| 合成一维 toy | `demo-generate`、`demo-predict`、`demo-rollout`、`demo-plan`、`demo-native` | 文件输出、线性预测、交互、采样规划、函数包装 | CPU 接口、生命周期、独立评价；不是研究基线 |
| Wan2.1 T2V 1.3B | `native-wan` | 文本生成视频 | Linux/CUDA 实际生成；完整参数历史验收与新版短 smoke 分开记录 |
| ModelScope T2V 1.7B | `native-modelscope` | 文本生成视频 | Windows 与 Linux 原生生成；新版 smoke 可播放 |
| DreamerV3 CartPole | `native-dreamer`、`predict-dreamer` | 记录序列的后验推理；固定动作下的未来状态预测 | 官方 RSSM、32 次训练更新的工程 checkpoint；未收敛 |
| RLVR-World language | `predict-rlvr_language` | Bytesized32 一步符号状态预测 | 作者 checkpoint、真实测试输入；三例精确转移评分均为 0 |
| RLVR-World video | `predict-rlvr_video` | 四帧 RT-1 上下文及动作条件下一帧预测 | 作者 RLVR checkpoint 与 tokenizer；单样本，训练成员关系未验证 |
| RENEW 受控试验 | `native-renew` | 2×2 2048 上比较六种反馈更新条件 | 真实 JAX 更新和 checkpoint；非完整论文复现 |
| OpenVLA 7B | `native-openvla` | 一张图像及指令输出七维动作 | 实际模型推理；没有动作执行或任务成功评价 |
| ManiSkill PushCube | `native-maniskill`、`maniskill-hold` | 官方运动规划器；保持关节位置的接口演示 | 真实 Panda 仿真；本身不含学习到的 WM |
| 小型状态 WM + ManiSkill | `state-wm-maniskill` | 采集、动作/无动作残差 MLP 训练、后果预测、可选录像 | 36 回合、3 个训练 seed、独立测试和独立评价 |
| TD-MPC2 + ManiSkill | `tdmpc2-pushcube-train`、`tdmpc2-pushcube-eval`、`tdmpc2-pushcube-predict` | 在线训练、MPPI 闭环控制、干预评测、潜空间预测 | 单训练 seed、102,400 次转移、三组各 10 个评测种子 |

源码研究笔记中出现的其他模型不等于已接入。没有对应配方或可调用适配器的模型不能直接提交运行。视频生成器、下一帧预测器、动作策略与仿真器也不能仅凭 JSON 格式统一就互相替换。

## 2. 所有模型共用的运行约定

### 2.1 存储与独立环境

在本文命令中，`WMS_ROOT` 表示你的项目存储根，`WMS_PROJECT_ROOT` 表示其中的 `deployments/wm-safetylab`。新建的源码、虚拟环境、缓存、资产、数据和结果分别放入该根目录下的 `repos/`、`envs/`、`cache/`、`assets/`、`datasets/`、`results/` 等目录。本文模型资源准备工具依赖这一目录布局：仓库应位于所选根目录的 `deployments/wm-safetylab` 下。

本项目现有 Windows 工作机器必须遵守 K 盘存储规则，运行或安装项目前加载项目根下的 `use-k-drive.ps1`，不在 C 盘创建项目文件。新机器则在自己的存储盘选择工作根，按 [部署指南](DEPLOYMENT_AND_USAGE.zh-CN.md) 加载 `scripts/windows/use-workspace.ps1` 并配置目录；不要求所有使用者都有 K 盘。现有目录应原位保留。

Linux 先将以下占位路径换成自己的实际存储根，确认仓库位于它的 `deployments/wm-safetylab` 中，再执行：

```bash
export WMS_ROOT=/your/storage/world-model
cd "$WMS_ROOT/deployments/wm-safetylab"
source scripts/linux/use-linux.sh
```

脚本保留显式设置的 `WMS_ROOT`，并设置 HF、Torch、pip、uv、CUDA、Matplotlib 的缓存目录。未显式设置时，服务部署默认使用仓库内 `.workspace`；这与部分历史模型准备工具要求的根目录布局不同，不能直接混用。以下安装示例均假定位于平台根目录，且 `python3.11` 是目标机器已准备的解释器。平台服务环境和模型环境分开；不同模型不要混装同一份 Transformers、JAX 或 CUDA 扩展。

```bash
# 示例：只创建独立环境；将 engine 换成本文相应模型名。
python3.11 -m venv "$WMS_ROOT/envs/engine"
PY="$WMS_ROOT/envs/engine/bin/python"
"$PY" -m pip install --upgrade pip
```

上游源码保持干净。适配、实验包装、机器配置和报告放在平台部署目录；不要在上游 `network.py` 或官方环境文件里直接打实验补丁。运行时设置 `PYTHONDONTWRITEBYTECODE=1` 可避免导入产生的缓存文件落入 checkout。

### 2.2 机器配置、预测 profile 与共享配方

有两种配置文件，不能混为一谈：

| 类型 | 谁读取 | 常见内容 |
|---|---|---|
| 普通原生 runtime | 注册表与后端 | `python`、`checkpoint_dir`、`source_dir`、`asset_dir`、`env`、`timeout_s` |
| 预测 profile | `world_model_predict` 包装器 | `python`、`config` 指向适配器配置、`request` 指向预测请求、`env` |

Dreamer 的 `predict-dreamer`、RLVR language/video 使用第二种；其他配方通常使用第一种。复制对应 `*.example.json` 为新的机器文件后，将所有占位路径替换为**运行服务机器可访问的实际路径**。JSON 不展开 `$WMS_ROOT`、`~` 或 shell 环境变量。模板中的占位路径不是可执行的默认部署。

共享配方的 `runtime.config_file` 相对配方文件解析；显式请求里的该路径相对平台根目录解析。预测 profile 的 `config`/`request` 相对 profile 文件解析。后端的权重、资产、数据等自定义路径通常直接交给 `Path` 或模型库，建议在机器文件里填写绝对路径，避免工作目录差异。

服务启动时可用一个本机 `native-configs.json` 映射文件选择机器配置：

```json
{
  "wan": "native/wan/runtime.json",
  "modelscope": "native/modelscope/runtime.json",
  "dreamer": "native/dreamer/runtime.json",
  "predict-dreamer": "native/world_model/dreamer.runtime.json",
  "renew": "native/renew/runtime.json",
  "openvla": "native/openvla/runtime.json",
  "maniskill": "native/maniskill/runtime.json",
  "predict-rlvr_language": "native/rlvr_language/runtime.json",
  "predict-rlvr_video": "native/rlvr_video/runtime.json",
  "tdmpc2": "native/tdmpc2/runtime.json"
}
```

映射值相对映射文件自身解析。仅保留已经准备的项，再以 `wms serve --native-configs native-configs.json` 启动服务。`state-wm-maniskill` 与两条 ManiSkill 配方共用 `maniskill` 键。

单次任务也可显式提供 `runtime.config_file`。运行配置合并优先级是：机器文件默认值 < 共享配方 < 本次显式配置。机器 runtime 的通用字段只有 `python/env/cwd/python_paths/timeout_s` 等由注册表读取，其余字段交给各适配器。老文件中的 `gpu_slots`、`wall_seconds` 不构成当前队列的显存预留或超时控制；实际预算使用 `timeout_s`。

`configured=true` 仅说明选择的机器 JSON 能够读取，**没有检查 GPU、依赖、权重、输入文件或源码完整性**。当前核心没有旧版模型专用预检系统。锁文件和报告中的 revision/hash 是溯源信息；除准备脚本或特定适配器明确执行的检查外，不应假设每次推理都重新核验所有资源。

### 2.3 提交、等待和读取结果

以下 `python` 应来自安装过平台的客户端环境。CLI 和 Python 客户端使用 `WMS_BASE_URL` 指向服务，例如服务本机的回环地址；远程机器应使用自己的连接配置。

```bash
export WMS_BASE_URL=http://127.0.0.1:8765
python -m wms recipes
python -m wms run demo-predict --wait
python -m wms run my-run.json --wait
python -m wms get-run RUN_ID
python -m wms logs RUN_ID
```

`my-run.json` 是相对当前终端的配置文件；其中资源路径依然由服务器解释。当前命令没有 `wms validate`、`--json` 或 `--wan-runtime/--renew-runtime` 这类旧参数。不要照历史文档调用已删除的 `wms.engines.world_model.worker`；使用本文的当前配方。

```python
import wms

submitted = wms.run({"recipe": "demo-rollout", "seed": 0, "steps": 8})
run = wms.wait(submitted["id"], timeout=3600)
if run["state"] != "completed":
    raise RuntimeError(run.get("error", run["state"]))
evaluation_id = run.get("automatic_evaluation_id")
if evaluation_id:
    evaluation = wms.wait(evaluation_id)
    if evaluation["state"] != "completed" or evaluation.get("evaluation_status") == "partial_failed":
        raise RuntimeError("独立评价没有完整成功")
print(run["id"], run.get("result", {}).get("outputs", []))
```

模型任务与评价任务有独立 ID；`run --wait` 不等待自动评价。只有 `metrics` 非空才会自动创建评价任务。默认 Wan、ModelScope、Dreamer、RLVR、OpenVLA、RENEW 和原生 ManiSkill 的 `metrics=[]`，不会自动给出性能或安全分数。可用 `wms.evaluate(run_id, metrics=[...])` 指定自己的评价函数，详见开发文档。`wms.wait` 客户端超时不会取消服务器任务，应查询原 ID，必要时调用 `wms.cancel`。

运行的 `result.outputs` 是可访问工件清单。使用 `GET /api/v1/runs/{id}`、`GET /api/v1/runs/{id}/evaluations` 和 `GET /api/v1/runs/{id}/files/{path}` 查看记录、评价和工件；文件路径来自清单，不需要硬编码服务器存储根。网页结果页可展示登记的 JSON、MP4、图片和 HTML。

### 2.4 快速核验顺序

1. `demo-predict` 确认客户端、服务与 worker。
2. 准备目标模型自己的环境和资源，检查机器文件路径。
3. 运行本章给出的短配置，查看原始输出及日志。
4. 再运行完整预算、固定数据分割和基线对照。
5. 单独等待评价，确认指标来源、版本、错误和统计口径。

接口 smoke 的完成不能代替预测准确性、任务成功或安全性评价。

## 3. CPU toy 与通用任务

不需要模型下载或 GPU。实现位于 [wms_demo.py](../examples/minimal/wms_demo.py)，配方位于 [最小注册表](../examples/minimal/registry.json)。

| 配方 | 输入与默认值 | 输出 |
|---|---|---|
| `demo-generate` | `inputs.prompt` | `sample.txt`，并非视频 |
| `demo-predict` | `inputs.observation=[0]`；`actions=[[0.1],[0.2]]` | 累加得到的预测状态 |
| `demo-rollout` | `goal=1`、可选 `initial`；`steps=8` | `trajectory.json`；独立 steps/return/cost_sum |
| `demo-plan` | `goal=1`、`steps=5`；动作范围 `[-0.2,0.2]`；默认 shooting | 规划后真实 toy 轨迹和独立指标 |
| `demo-native` | `inputs.message="hello"`；`steps=1`；可选 `params.delay` 或 `fail` | `echo.json`；可用于失败/取消流程演示 |

`LinearModel` 是解析线性动力学，`PointEnv` 的转移是状态加动作。`cost=1` 仅表示合成状态绝对值超过 2，不是物理危险标签。rollout 的策略目标来自 `model.config.goal`，而环境目标来自 `inputs.goal`；修改目标时应同时设置两者。

```json
{
  "recipe": "demo-plan",
  "algorithm": "cem",
  "inputs": {"initial": 0, "goal": 1},
  "steps": 5
}
```

注册表已提供 shooting（horizon 3、8 candidates）和 CEM（horizon 3、population 8、elites 2、iterations 2）。这是规划器调用示例，不是论文模型训练。

## 4. Wan2.1 文本生成视频

### 4.1 资源与安装

- 上游：[Wan2.1](https://github.com/Wan-Video/Wan2.1/tree/9737cba9c1c3c4d04b33fcad41c111989865d315)，commit `9737cba9c1c3c4d04b33fcad41c111989865d315`。
- 权重：[Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/37ec512624d61f7aa208f7ea8140a131f93afc9a)，revision `37ec512624d61f7aa208f7ea8140a131f93afc9a`。
- 已验环境：Linux、Python 3.11、Torch 2.7.0+cu128、torchvision 0.22.0、FlashAttention 2.8.3。当前后端需要 CUDA。

在独立环境中安装上述 Torch/torchvision，并按锁定上游依赖准备其余包。FlashAttention wheel 必须匹配 CPython 3.11、Torch 2.7、CUDA 12 与 C++ ABI；现有验收使用 Linux x86_64 的 `flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl`。这套原生依赖没有完成 Windows 等价验收。

源码可使用 [fetch_source.py](../native/wan/fetch_source.py) 下载固定归档到项目 `repos/`，也可干净 clone 后 checkout 指定 commit。下载完整固定权重快照；必须包含扩散模型、`models_t5_umt5-xxl-enc-bf16.pth`、`Wan2.1_VAE.pth`、`config.json` 和 `google/umt5-xxl/` tokenizer，仅下载 1.3B 扩散网络不够。

从 [runtime.example.json](../native/wan/runtime.example.json) 建立机器 runtime，填写 `python/source_dir/checkpoint_dir/cuda_visible_devices`。若保留 `weights_lock`，该 JSON 必须确实存在；也可直接在 runtime 的 `weights` 对象登记 repository/revision。旧 `prepare_linux.py` 依赖特定历史元数据目录和环境布局，不是新主机的一键安装器；新部署应自行保存固定 revision 下载清单、哈希与环境 freeze，再填写 runtime。

### 4.2 配方参数与运行

`native-wan` → `wms.engines.wan.native:wan_generate`。所有生成参数在 `inputs.generation`，随机种子在顶层 `seed`。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `inputs.prompt` | `A red cube on a wooden table.` | 文本条件 |
| `size` | `[832,480]` | 宽、高 |
| `frame_num` | 81 | 输出帧数；按上游时序压缩要求选择，smoke 使用 17 |
| `sample_solver` | `unipc` | 官方采样器 |
| `sampling_steps` | 50 | 去噪步数 |
| `shift` / `guide_scale` | 8.0 / 6.0 | 时间步偏移 / CFG |
| `offload_model` / `t5_cpu` | true / true | 模型卸载 / 文本编码器留 CPU |
| `prompt_extend` | false | 当前后端不读取此字段，没有提示扩展流程 |

```json
{
  "recipe": "native-wan",
  "name": "Wan 短 smoke",
  "seed": 0,
  "timeout_s": 1800,
  "inputs": {"generation": {"frame_num": 17, "sampling_steps": 4}}
}
```

保存配置后用 `python -m wms run my-run.json --wait`。嵌套对象会与配方合并，不必重复全部默认字段。`steps=8` 是历史通用字段，不控制 Wan 去噪或帧数。FPS 来自官方配置，当前为 16，不是 `inputs.generation.fps`。

产物为 `video.mp4` 与 `runtime_report.json`（时间、峰值 Torch 分配、版本、负提示和实际 FPS）。新版短运行记录为 17 帧/4 步，在 RTX 4090 的一次记录约 68.7 秒、峰值分配约 8.42 GB；这是观测值，不是完整配置的显存或速度保证。历史完整验收是 832×480、81 帧、50 步，不能把短 smoke 算作完整参数重验。

当前没有 I2V、多 GPU、训练、物理动作条件或安全评分配方。

## 5. ModelScope 文本生成视频

### 5.1 资源与配置

使用 [ali-vilab/text-to-video-ms-1.7b](https://huggingface.co/ali-vilab/text-to-video-ms-1.7b/tree/8227dddca75a8561bf858d604cc5dae52b954d01)，固定 revision `8227dddca75a8561bf858d604cc5dae52b954d01`；加载的是 Diffusers `TextToVideoSDPipeline`。

独立环境采用 Torch 2.7.0+cu128、Diffusers 0.35.1、Transformers 4.49.0、Accelerate 1.10.1、NumPy 1.26.4；完整版本见 [requirements-native.lock](../native/modelscope/requirements-native.lock)。含 `+cu128` 的 Torch 包需从对应 PyTorch 索引取得，不能只靠默认 PyPI 安装整份锁文件。

[download.py](../native/modelscope/download.py) 下载固定 FP16 safetensors、配置及 tokenizer，并生成 `weights-lock.json`。在新部署的独立解释器执行 [configure.py](../native/modelscope/configure.py) 可写 `runtime.json` 和环境信息；它会写本机配置，不要在已有部署上当只读检查运行。也可从 [runtime.example.json](../native/modelscope/runtime.example.json) 手工建立新的配置目录，填写 `python/checkpoint_dir/weights/cuda_visible_devices`。

### 5.2 运行与输出

| `inputs.generation` 字段 | 默认值 |
|---|---:|
| `height` / `width` | 256 / 256 |
| `num_frames` | 16 |
| `num_inference_steps` | 25 |
| `guidance_scale` | 9.0 |
| `fps` | 8 |

```json
{
  "recipe": "native-modelscope",
  "seed": 0,
  "inputs": {
    "prompt": "A red cube on a wooden table.",
    "generation": {"num_frames": 8, "num_inference_steps": 4}
  }
}
```

后端固定 FP16、DPM multistep scheduler、CPU offload、VAE slicing 和 UNet forward chunking；需要 CUDA，即使主要权重会卸载到 CPU。输出 `video.mp4` 和 `runtime_report.json`。新版 256×256、8 帧/4 步的实跑约 6.56 秒、峰值 Torch 分配约 3.03 GB。文本视频质量、安全、物理一致性和闭环控制均未由默认配方评价。

## 6. DreamerV3：两种不同的离线任务

### 6.1 准备源码、数据和工程 checkpoint

使用 [danijar/dreamerv3](https://github.com/danijar/dreamerv3/tree/e01491fad6434b2245a3b8ca201dd7faedcc458c) commit `e01491fad6434b2245a3b8ca201dd7faedcc458c`，以及 [NathanGavenski/CartPole-v1](https://huggingface.co/datasets/NathanGavenski/CartPole-v1/tree/523ca21d86ef6844bad6e520cfa364ad132f1ed1) 的固定 `teacher.jsonl`。没有下载一个“已收敛的 Dreamer 策略”；本项目训练的是小型工程 checkpoint。

独立 CPU 环境使用 [requirements-linux.lock](../native/dreamer/requirements-linux.lock)，核心为 JAX/JAXlib 0.4.33、NumPy 1.26.4、Elements 3.19.1、Ninjax 3.5.1、Optax 0.2.3。

```bash
PY="$WMS_ROOT/envs/dreamer/bin/python"
"$PY" -m pip install -r native/dreamer/requirements-linux.lock
"$PY" native/dreamer/prepare_assets.py --world-root "$WMS_ROOT"
"$PY" native/dreamer/train_offline.py \
  --asset-lock native/dreamer/source-lock.json \
  --output "$WMS_ROOT/models/dreamer-cartpole-engineering-new" --updates 32
```

脚本生成 `checkpoint.npz`、`agent-config.json`、`counters.json`、`inference-sequence.json`、训练记录、数据划分与 `weights-lock.json`。输出目录要新建；旧 checkpoint 不会被静默覆盖。前两个记录回合训练，第三个回合留作推理。默认 32 次更新不足以证明收敛。

从 [Dreamer runtime 模板](../native/dreamer/runtime.example.json) 生成普通 runtime，引用这一 checkpoint 目录及其生成的 `weights-lock.json`。推理强制 CPU；不会创建 CartPole 仿真器。

### 6.2 `native-dreamer`：后验推理

```json
{"recipe":"native-dreamer","inputs":{"sequence_steps":8},"seed":0}
```

默认 `sequence_steps=32`，读取 checkpoint 内的记录序列。官方 encoder/RSSM 吸收真实历史观测与前一记录动作，输出重建状态、latent 范数和 actor 预测动作。**这些 actor 动作没有生成输入序列，也没有被执行。** `id/checkpoint_class` 是描述信息，不能用它们切换成其他模型；checkpoint 由 runtime 选择。

输出 `sequence.json`、`trajectory.json`、`runtime_report.json`。该路径不是未来状态预测误差实验，不能把后验重建误差当 forecast 指标。

### 6.3 `predict-dreamer`：固定未来动作的开环预测

复制 [预测 profile 模板](../native/world_model/dreamer.runtime.example.json)，让 `config` 指向普通 Dreamer runtime，`request` 指向自己的请求 JSON。可从 [请求模板](../native/world_model/dreamer.request.example.json) 起步；示例数值仅演示格式，不是测试集证据。

请求字段如下：

| 字段 | 约束 |
|---|---|
| `model_id` | `dreamerv3-cartpole-predict` |
| `horizon` | `steps=1..64`，`unit="environment_step"` |
| `inputs.context_observations` | 1..128 个四维 CartPole 状态，单回合上下文 |
| `inputs.context_actions` | 长度为上下文状态数减 1，整数 0 或 1 |
| `inputs.actions` | 长度等于 horizon，整数 0 或 1 |
| `parameters` | `{}`，不提供自由采样参数 |

未来动作第 t 项从上下文最后状态开始对应第 t 次未来转移。四个状态坐标为车位置/速度、杆角/角速度，分别使用 m、m/s、rad、rad/s。

```bash
python -m wms run predict-dreamer --wait
```

输出 `prediction.json`、`trace.json`、上下文与未来 latent/logits 数组、`future_decoded_state.npy`、`future_reward.npy`、`future_continue.npy` 等。后两者分别是奖励和折扣 continuation 输出，不是校准的物理风险或安全概率。一个 seed 对应一条随机 latent 轨迹；没有在预测过程中读取未来观测，也没有建立预测质量基准。

## 7. RLVR-World language：一步符号状态预测

### 7.1 资源与安装

资源均固定到以下版本：

| 资源 | 仓库与 revision |
|---|---|
| 源码 | [thuml/RLVR-World](https://github.com/thuml/RLVR-World/tree/e1b8b6f40ca0696919ce9b8dbe965c4153bcf5f1)，`e1b8b6f40ca0696919ce9b8dbe965c4153bcf5f1` |
| 作者模型 | [bytesized32-world-model-rlvr-task-specific-reward](https://huggingface.co/thuml/bytesized32-world-model-rlvr-task-specific-reward/tree/153dd625edde0fef09bad468235c14019d91ce3f)，`153dd625edde0fef09bad468235c14019d91ce3f` |
| 数据 | [bytesized32-world-model-cot](https://huggingface.co/datasets/thuml/bytesized32-world-model-cot/tree/f80652121da810c8f856e1a8208f5fc60c2f8b78)，`f80652121da810c8f856e1a8208f5fc60c2f8b78`，只需 `test_state_difference.parquet` |

实测独立 Linux 环境为 Torch 2.7.0+cu128、Transformers 4.51.3、Tokenizers 0.21.4、NumPy 1.26.4、Accelerate 0.34.2、HF Hub 0.36.2、PyArrow 20.0.0、json-repair 0.44.1、dirtyjson 1.0.8。模型 safetensors 约 3.55 GB；实测推理峰值 Torch 分配约 4.09 GB，不含其他进程及全部设备开销。

准备干净源码到构建器约定的 `repos/RLVR-World-lang-e1b8b6f-linux`，使用 Hugging Face 客户端下载上表的固定资源并创建本机清单。以下命令用于新部署，环境名称应与构建器默认布局一致：

```bash
PY="$WMS_ROOT/envs/rlvr-language-20261008/bin/python"
git clone https://github.com/thuml/RLVR-World.git "$WMS_ROOT/repos/RLVR-World-lang-e1b8b6f-linux"
git -C "$WMS_ROOT/repos/RLVR-World-lang-e1b8b6f-linux" checkout --detach e1b8b6f40ca0696919ce9b8dbe965c4153bcf5f1
"$PY" - <<'PYTHON'
import hashlib, json, os
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
root = Path(os.environ['WMS_ROOT'])
model_id = 'thuml/bytesized32-world-model-rlvr-task-specific-reward'
model_rev = '153dd625edde0fef09bad468235c14019d91ce3f'
dataset_id = 'thuml/bytesized32-world-model-cot'
dataset_rev = 'f80652121da810c8f856e1a8208f5fc60c2f8b78'
model = root / 'models/rlvr-language-task-specific-153dd625'
snapshot_download(model_id, revision=model_rev, local_dir=model)
dataset = Path(hf_hub_download(dataset_id, 'test_state_difference.parquet',
    repo_type='dataset', revision=dataset_rev,
    local_dir=root / 'datasets/rlvr-language-f8065212'))
files = []
for path in [*(p for p in model.rglob('*') if p.is_file() and '.cache' not in p.parts), dataset]:
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    files.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest})
manifest = {'model_id': model_id, 'model_revision': model_rev,
    'dataset_id': dataset_id, 'dataset_revision': dataset_rev,
    'upstream_source_commit': 'e1b8b6f40ca0696919ce9b8dbe965c4153bcf5f1',
    'endpoint': os.environ.get('HF_ENDPOINT', 'https://huggingface.co'), 'files': files}
target = root / 'deployments/wm-safetylab/native/rlvr_language/resources.server.json'
target.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
PYTHON
"$PY" native/rlvr_language/run_smoke.py --prepare-only --cases 3
```

`--prepare-only` 不推理，但会生成/重写机器 profile、adapter 配置、manifest、三个请求及独立参考答案。应在新部署或备份后的独立准备阶段运行。旧部署使用的 `prepare_resources.py/prepare_server.sh` 含历史下载/机器约定，没有作为公共安装入口发布；新机器按上述版本自行建立独立环境。若所在网络需要镜像，可在下载前显式配置 `HF_ENDPOINT`，仍保留同一官方仓库和 revision。

手工配置可用 [runtime.example.json](../native/rlvr_language/runtime.example.json) 与 [adapter.config.example.json](../native/rlvr_language/adapter.config.example.json)。适配器读取 `model_dir/source_dir`；`resource_manifest` 可选，用于附带来源元数据。profile 不需要预先存在的 manifest 才能推理，`describe()` 会产生模型描述。

### 7.2 请求与运行

用数据构建器生成的 `request.example.json`，不要把 ground truth 放进模型提示。

- `model_id="rlvr-language-bytesized32-task-specific"`。
- `horizon={"steps":1,"unit":"transition"}`。
- `inputs.messages` 为实际上下文的 user/system 消息；`current_state` 包含 `game_state` 列表；`action` 必须出现在提示中；`dataset_reference` 可登记来源。
- `parameters.max_new_tokens` 为 1..4096，默认 4096；`do_sample=false`。超过 7192 prompt tokens 会报错，不隐式截断。

```bash
python -m wms run predict-rlvr_language --wait
```

若需要本次覆盖 horizon、parameters 或输入，放在平台任务的 `inputs` 对象内，例如 `inputs.parameters={"max_new_tokens":1024,"do_sample":false}`。这里是预测包装器的请求字段，并非平台通用 `params.max_new_tokens`。保留 profile 的基础请求时，未覆盖的字段沿用基础值；顶层平台 `seed` 会覆盖 profile 请求 seed。

输出包括 `prompt.txt`、`raw_output.txt`、`generated_token_ids.json`、`partial_state.json`、`predicted_state.json`、`diagnostics.json` 与 `prediction.json`。原生模型产生 `modified/removed/score` 状态差分，然后用官方恢复函数重建状态。

应检查 `diagnostics.parse_status/strict_json_valid/repair_applied/token_limit_reached`。当前实现会保存解析错误的 diagnostics，并可能仍返回一个 completed bundle；**平台 completed 不能单独证明产生了合法预测状态**。精确转移评分需要模型外的参考答案及官方 scorer；默认配方没有自动评分。

已有 smoke 从 2,954 个测试记录中按最短输入选三例（2153、2159、2163），选择时未使用预测结果；三例严格 JSON 均有效，但官方精确转移分数全部为 0，task-specific reward 约 0.0904—0.0936。它证明作者 checkpoint 真正执行且产生错误，不是代表性 benchmark，也未验证分布式 verl/vLLM 等价性。

## 8. RLVR-World video：RT-1 下一帧预测

### 8.1 资源、环境和样本

源码与语言模型共用上述 RLVR-World commit，但应使用独立原生环境。权重有两组：

- [rt1-world-model-single-step-rlvr](https://huggingface.co/thuml/rt1-world-model-single-step-rlvr/tree/ba6dae4049067c703f236f709720f6ff216c7e5f)，revision `ba6dae4049067c703f236f709720f6ff216c7e5f`，模型约 482 MB。
- [rt1-frame-tokenizer](https://huggingface.co/thuml/rt1-frame-tokenizer/tree/a41f1ad0fc30da2d24bfc346dee65d2f70703bfd)，revision `a41f1ad0fc30da2d24bfc346dee65d2f70703bfd`，约 418 MB。

还需要上游 `vid_wm/ivideogpt/configs/vgpt/frac_action_ranges.pth`。下载完整的固定快照和配置，不能拿普通 iVideoGPT base 权重替代作者 RLVR 权重。

```bash
PY="$WMS_ROOT/envs/rlvr-video/bin/python"
"$PY" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0 torchvision==0.22.0
"$PY" -m pip install -r native/rlvr_video/requirements-inference.txt
```

小依赖锁包含 Transformers 4.44.2、Diffusers 0.27.0、NumPy 1.26.4、HF Hub 0.25.2。tokenizer 使用官方 BF16 autocast，Llama 模型使用 FP32。与语言模型的 Transformers 版本不同。

可直接提供满足下节协议的真实上下文 NPZ。复现已有 smoke 时，数据来源为 Google 的 `fractal20220817_data/0.1.0` shard 00000；读取其第一条完整 TFRecord（字节 0..3,477,492），而非下载全量数据。[prepare_sample.py](../native/rlvr_video/prepare_sample.py) 要求数据目录已有 `episode-00000.tfrecord`、该数据集的 `dataset_info.json` 和 `features.json`，会核对 TFRecord 长度与 CRC，再生成 `context.npz`、独立 `reference.npz` 和 provenance：

```bash
"$PY" native/rlvr_video/prepare_sample.py --data-root "$WMS_ROOT/datasets/rlvr-video-smoke"
```

原始数据 shard URL 及 metadata 所在前缀在准备脚本中。应记录原始下载字节范围和哈希。已有 episode 有 115 帧，按作者 item 0/seed 0 规则选上下文 44—47、参考帧 48。作者 checkpoint 没有训练 episode 成员清单，因此标记 `split_unverified`，不能称已验证 held-out 泛化。

使用 [runtime.example.json](../native/rlvr_video/runtime.example.json)、[adapter.config.example.json](../native/rlvr_video/adapter.config.example.json) 创建机器配置，设置源码、两组模型、数据根、`device`、`cpu_threads`。需要自己生成指向真实 context 的请求；仓库中的历史请求含部署路径，不适合直接迁移。

### 8.2 请求语义与运行

`model_id="rlvr-video-rt1-single-step-rlvr"`，`horizon={"steps":1,"unit":"video_frame"}`，`parameters={}`。`inputs.context_npz` 指向只包含以下数组的文件：

| 数组 | 类型与形状 |
|---|---|
| `image` | `uint8[4,H,W,3]`；H/W 在 16..4096 内 |
| `action` | 有限数值 `[4,13]`，与四帧逐帧对齐 |

动作顺序固定为 `base_displacement_vector[2]`、`base_displacement_vertical_rotation[1]`、`gripper_closedness_action[1]`、`rotation_delta[3]`、`terminate_episode[3]`、`world_vector[3]`。第四帧动作也是条件。相同维数的别的机器人动作不自动兼容此协议。

```bash
python -m wms run predict-rlvr_video --wait
```

后端缩放到 256×320，四帧视觉/动作前缀，固定 `temperature=1/top_k=100/top_p=1`，生成 321 token（320 个视觉 token 加末 token）。输出 `forecast.npy`（float32 `[1,256,320,3]`）、`predicted_frame.png`、`context_and_prediction.mp4`、`token_trace.npz`、`prediction.json`。五帧 MP4 前四帧来自上下文，最后一帧为预测；4 FPS 仅为展示，不是机器人控制频率。

独立像素评价使用 [evaluate_smoke.py](../native/rlvr_video/evaluate_smoke.py)，其 `--prediction-dir` 指向本次预测输出目录，`--data-root` 指向含 reference 的目录，`--output-dir` 使用新的评价目录。当前平台任务直接把 forecast 放在 run 的 output 中，不再使用旧 worker 的额外 `forecast/` 子目录。

已有单样本结果：模型 MAE 0.02171、PSNR 29.26 dB；复制最后观测帧基线 MAE 0.008995、PSNR 38.79 dB，模型在该样本上误差更大。新版推理峰值 Torch 分配约 1.58 GB。像素误差不是安全指标。

当前代码读取同一份 context 字节快照并检查数组形状；并不强制校验历史请求中的 `context_sha256`、`dataset_root` 路径边界或每次源码 clean 状态。需要精确复现时，应在准备与独立审计阶段自行核验这些资源，而不是仅信任 manifest 中写出的版本。

## 9. RENEW：2×2 2048 反馈更新机制试验

### 9.1 部署与实际范围

使用 [FlyingWorkshop/RENEW](https://github.com/FlyingWorkshop/RENEW/tree/8f6e1ebf66a0200cc2015226cdd7d70900fc0cb7) 的 `LatentCNNNetwork`、2048 辅助函数和 checkpoint 方法，commit `8f6e1ebf66a0200cc2015226cdd7d70900fc0cb7`。本实验从本次监督预热权重开始，不要求外部预训练 checkpoint。

独立 Linux CPU 环境的已验核心版本：JAX 0.9.0.1、Flax 0.12.4、Jumanji 1.1.1；按锁定上游依赖准备其余包，并补齐 `requests==2.32.5`（Jumanji 导入链需要）。训练进程关闭 CUDA，最多限制到 8 个 CPU 核。

安装完成后，用独立解释器生成当前机器配置和完整 freeze：

```bash
"$WMS_ROOT/envs/renew/bin/python" native/renew/configure_linux.py \
  --source "$WMS_ROOT/repos/RENEW" --output "$WMS_PROJECT_ROOT/native/renew/local-new"
```

该配置工具会验证指定 commit 和干净源码，拒绝覆写不同的已有锁；这属于准备阶段检查。将映射的 `renew` 指向生成的 runtime，或从 [runtime.example.json](../native/renew/runtime.example.json) 填写 `python/source_dir/source_commit`。

### 9.2 输入和输出

| `inputs` 参数 | 默认值 | 实际作用 |
|---|---|---|
| `warmup_steps` | 200 | 共同模型 q0 的监督预热次数 |
| `update_steps` | 40 | 非 Frozen 条件的更新次数 |
| `batch_size` | 32 | 预热与更新批大小 |
| `risk_threshold` | 0.15 | 固定接受/暂缓阈值 |
| `methods` | frozen、bt、renew_style、mle、dpo、full_group | 运行哪些已实现条件 |
| `board_size/num_candidates/num_tiles` | 2 / 3 / 5 | 当前后端将这些协议值固定为 2×2、3 候选、5 类；改 JSON 不改变后端 |
| `id/max_steps/scope` | 描述字段 | 当前训练后端不以它们控制实际协议或预算 |

```json
{"recipe":"native-renew","seed":0,"inputs":{"warmup_steps":20,"update_steps":5,"batch_size":8}}
```

这是短集成配置；完整默认配方直接 `python -m wms run native-renew --wait`。输出 `input.json`、`support.json`、`feedback.json`、`predictions.json`、`runtime_report.json`、`checkpoints.zip`。

后端枚举 625 个棋盘的完整支持，使用真实环境转移构造上下文，保存逐条件分布 q 与真实分布 p，测量 TV、NLL、Brier、非法状态质量和固定决策。代理事件是“新块出现在位置 0”，不是物理危险。

六种条件的信息量不同：Frozen 不更新；BT/DPO 使用 pair01 投影；RENEW-style 使用三候选的最佳者和严格劣者；MLE 直接获得真实后继；full_group 还知道完整反馈通道。不能把它们解释成只换损失函数的等信息消融。当前没有执行第七种 selection-aware objective，也没有主动查询、机器人或语言 WM 闭环。

新版已完成默认接口下的真实短运行，旧机制实验有完整预算证据。当前 `metrics=[]`，报告中的研究测量不等于平台自动安全评价；各方法全部暂缓或部分分数变差也可能是有效结果。

## 10. OpenVLA：单图像到动作

使用 [openvla/openvla-7b](https://huggingface.co/openvla/openvla-7b/tree/47a0ec7fc4ec123775a391911046cf33cf9ed83f)，revision `47a0ec7fc4ec123775a391911046cf33cf9ed83f`。下载包括 config、处理器、tokenizer、模型 Python 文件、safetensors index 及全部分片的完整 checkpoint；动作反归一化统计位于 `config.json`，不要求额外 `dataset_statistics.json`。

已实测环境使用 Torch 2.2.0+cu121、Transformers 4.40.1，另需 timm、Accelerate、safetensors、Pillow、Tokenizers、NumPy 等匹配上游依赖。**发布仓库没有一份完整、可搬移的 OpenVLA 原生环境锁或通用安装脚本**；按固定 checkpoint 的依赖要求建立独立环境，记录实际 freeze，再做单次推理核验。不要把语言或视频模型的 Transformers 版本直接复用给它。

从 [runtime.example.json](../native/openvla/runtime.example.json) 建立本机配置。最小必需项为 `python/checkpoint_dir`，GPU 由 `cuda_visible_devices` 选择。模板里的机器路径和 sidecar 引用都要替换；如不使用输入锁，可删除 `input_lock` 并直接在 runtime 的 `input` 或本次 `inputs` 中提供：

```json
{
  "recipe": "native-openvla",
  "seed": 0,
  "inputs": {
    "image_path": "/path/on/execution-host/observation.png",
    "instruction": "put the object in the container",
    "unnorm_key": "bridge_orig"
  }
}
```

这里 `/path/on/execution-host/` 是待替换占位符。默认 `action_semantics` 来自配方，标为 dataset-defined；只有确知目标机器人映射时才能进一步注明实际坐标系和单位。

输入优先级为 `params.input_defaults` < runtime 的 `input` < 本次 `inputs`。后端使用 BF16、eager attention、greedy `predict_action`，没有量化或 CPU offload。checkpoint 的自定义模型代码通过 `trust_remote_code=True` 从本地加载；固定可信模型目录并在安装时核对来源，当前代码没有旧版自动 Python 文件审查 gate。

输出 `action.json`、`runtime_report.json`，以及结果 `data.action`。动作严格为七个有限数值，顺序 `x,y,z,roll,pitch,yaw,gripper`。默认反归一化键为 `bridge_orig`；七维输出不等于八维 Panda 关节控制命令。

新版验收已修复官方动作后缀 token 与 attention mask 长度对齐的问题；一次 RTX 4090 推理耗时约 0.51 秒（不含加载）、峰值 Torch 分配约 15.48 GB。`executed=false`、`executed_on_robot=false`、`task_success_evaluated=false`：没有机器人、LIBERO 或 ManiSkill 控制结果。

## 11. ManiSkill 仿真与官方规划器

### 11.1 准备环境

已验证的是 Linux、Python 3.11、ManiSkill 3.0.1、SAPIEN 3.0.3、Torch 2.7.0+cu128、Gymnasium 1.4.0、NumPy 1.26.4、mplib 0.1.1，完整环境参考 [requirements-native.lock](../native/maniskill/requirements-native.lock)。先安装 CUDA Torch，再安装锁定依赖；资产目录必须位于项目存储根。CPU 物理仿真仍使用 GPU 渲染，需要正常的 NVIDIA Vulkan ICD。

在新目录生成配置：

```bash
mkdir -p "$WMS_ROOT/assets/maniskill"
"$WMS_ROOT/envs/maniskill/bin/python" native/maniskill/configure.py \
  --output-dir "$WMS_PROJECT_ROOT/native/maniskill/local-new" \
  --asset-dir "$WMS_ROOT/assets/maniskill" --gpu-device 0
```

配置工具不下载依赖或资产；它检查 Linux/ManiSkill 版本并保存锁。也可填写 [runtime.example.json](../native/maniskill/runtime.example.json)。若自动选择到错误的 Vulkan ICD，在 runtime 的 `env.VK_ICD_FILENAMES` 指定目标机器真实 NVIDIA ICD；不要复制另一台机器的驱动路径。

### 11.2 `native-maniskill`：真实官方运动规划示例

固定协议为 `PushCube-v1/panda/pd_joint_pos/physx_cpu/official_motion_planner`。更换这些字符串会被后端拒绝；新增任务应建立新适配器。

```json
{"recipe":"native-maniskill","seed":0,"steps":500}
```

`steps` 是真正生效的回合上限，默认 500；它覆盖 `inputs.max_steps`。后端只执行一个回合、不按成功与否筛选重试，保存 reset 初始画面与每步后画面。官方规划器读取特权仿真状态。

动作是 7 个绝对关节目标（rad）加 1 个归一化夹爪命令。输出 `input.json`、`trajectory.json`、`runtime_report.json`、`video.mp4`。report 区分 `task_success`、`success_once`、规划器状态、终止/截断和预算。成功不代表碰撞或力约束合格。

### 11.3 `maniskill-hold`：通用环境接口

```bash
python -m wms run maniskill-hold --wait
```

默认 8 步，`PandaPushCubeEnv` 返回 28 维状态：9 qpos + 9 qvel + 7 cube pose + 3 goal position。固定策略保持初始七个关节角并打开夹爪，输出普通 `trajectory.json`，不自动录制视频。它验证 `reset/step/close` 与 `act` 的接口，不训练策略或 WM。不要把这里的 28 维状态与下一节的 24 维或 TD-MPC2 的实际 35 维互换。

## 12. 小型状态 WM + ManiSkill：采集、训练、预测与录像

此实验复用上一节的 ManiSkill 原生环境，加 Torch CPU 训练；录像/图表额外需要已有环境中的 ImageIO/FFmpeg、Matplotlib。配方 [state-wm-maniskill](../native/state_wm_maniskill/recipes.json) 指向 `wms.engines.state_wm_maniskill.native:run`。

### 12.1 参数

参数默认在 `params`，本次 `inputs` 里的同名字段会覆盖它们。

| 字段 | 默认值 | 说明 |
|---|---|---|
| `episodes` | 36 | 至少 6，按 episode 分训练/验证/测试 |
| `max_episode_steps` | 160 | 每个采集回合上限 |
| `action_noise_std_rad` | 0.003 | 七关节绝对目标噪声，允许 0..0.05 rad |
| `data_seed_start` / `split_seed` | 0 / 20261010 | 环境种子起点 / episode 分割 |
| `training_seeds` | `[0,1,2]` | 有动作与无动作模型各训练这些 seed |
| `epochs` / `hidden_width` / `batch_size` | 400 / 128 / 256 | 验证集选 checkpoint |
| `lr` / `cpu_threads` | 0.001 / 4 | Adam 学习率 / CPU 线程 |
| `horizon` | 5 | 递归预测长度；同时保留一步预测 |
| `record_video` / `visualization_seed` | false / 0 | 录测试回合并生成对应模型 seed 的可视化 |
| `dataset_dir` | 不设置 | 复用旧目录里的 `dataset.npz/dataset.json`，复制后重新训练 |

```json
{"recipe":"state-wm-maniskill","inputs":{"record_video":true,"visualization_seed":0}}
```

也可运行仓库现成配置：

```bash
python -m wms run examples/state-wm-maniskill-video.json --wait
```

`record_video=true` 不能与 `dataset_dir` 同用，未记录的历史画面不能从数据文件伪造。顶层通用 `steps` 不替代 `max_episode_steps/epochs/horizon`。环境和模型训练种子以表中明确字段为准。

### 12.2 数据、产物和指标

24 维状态为 9 qpos + 9 qvel + 3 cube xyz + 3 TCP xyz；有动作模型输入实际限幅后的 8 维命令。机械臂仍由固定官方规划器执行，学习到的模型只离线预测。

产物为 `dataset.npz/dataset.json`、`predictions.npz`、`training.json`、`experiment.json`、六个 `.pt` checkpoint（默认两模型×三 seed）。自动独立评价入口为 `wms.engines.state_wm_maniskill.metrics:measure`，输出 `state_wm_metrics.json` 及平台 `metrics.json`。

录像额外产生测试回合原片、`simulation.mp4`、`comparison.mp4`、两组逐帧 CSV、PNG/PDF 曲线、`recordings.json`、`visualization.json` 和 `visualization.html`。视频来自真实仿真渲染，不是 WM 生成的视频。目标帧 k 的 H 步预测从 k−H 真值初始化，随后递归使用预测状态与记录动作；最初 H 帧没有对应预测。

完整既有实验为 36 回合/2,217 转移，划分 24/6/6 回合；3 个训练 seed。先对每回合内部平均、再对回合等权平均、最后平均 seed：

| 模型 | 一步 cube 误差 mm | 五步 cube 误差 mm | 一步 TCP 误差 mm | 五步 TCP 误差 mm |
|---|---:|---:|---:|---:|
| 状态保持 | 1.7196 | 8.3913 | 5.1699 | 25.8507 |
| 无动作 MLP | 0.6458 | 2.7862 | 1.1294 | 5.9387 |
| 动作条件 MLP | 0.5310 | 1.9325 | 0.9412 | 3.6548 |

有动作/无动作模型参数量分别为 23,832/22,808，不是严格等参数对照。36 个采集回合都曾成功，数据分布较容易。该结果支持这一分布上的后果预测差异，不支持安全防护、概率校准、反馈更新或跨任务泛化结论。

## 13. TD-MPC2 + ManiSkill：完整可执行研究流程

详细实现说明见 [TD-MPC2 教程](TD_MPC2_PUSHCUBE_GUIDE.zh-CN.md)，本节给出从资源到复用的最短路径。

### 13.1 环境准备

使用 [ManiSkill 官方 TD-MPC2 baseline](https://github.com/mani-skill/ManiSkill/tree/62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3/examples/baselines/tdmpc2)，commit `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`。复用官方网络、损失、Buffer、优化器和 MPPI；不是仅用同名网络重新写一套算法。

建立与 ManiSkill 相同的基础 GPU 栈，再在 TD-MPC2 独立环境安装 `hydra-core==1.3.2`、`hydra-submitit-launcher==1.2.0`、`termcolor==2.4.0`、`tensordict==0.8.3`、`torchrl==0.8.1`、`moviepy==1.0.3`、`wandb==0.19.11`。虽然安装 wandb，当前运行配置关闭在线跟踪。

已有服务器采用 `.pth` 引用 ManiSkill base 的 overlay。这个环境依赖 base，不能只复制 overlay venv 就宣称完成迁移。新主机可准备完整独立环境，也可明确保留相同 Python ABI 的 base+overlay，并保存两边版本。

从 [runtime.example.json](../native/tdmpc2/runtime.example.json) 填写 `python/source_dir/asset_dir/physx_gpu_library/gpu_device/cpu_threads/env/timeout_s`。PhysX GPU 库取 SAPIEN `105.1-physx-5.3.1.patch0` 对应的 `libPhysXGpu_64.so`，放在项目缓存；不依赖首次运行自动写入用户主目录。已有机器遇到 CUDA stub 与 Vulkan ICD 问题，是机器特定排障，不能在新机盲目复用私有 driver 链接。

[prepare_server.sh](../native/tdmpc2/prepare_server.sh) 记录了上述安装和 PhysX 下载过程，但含特定机器路径、base 路径和驱动处理。它是已部署环境的过程脚本，跨主机使用前应在部署目录另存副本并修改配置；不要直接在其他环境执行。

### 13.2 状态、动作与回合

当前固定 `PushCube-v1/panda/state/pd_joint_delta_pos/normalized_dense/physx_cuda`。观测直接取环境 state 向量，已验证部署实际为 35 维；动作实际为 8 维、范围 `[-1,1]` 的归一化增量控制命令，不是绝对关节角 rad。

每回合固定 50 步，成功不提前结束，显式 reset。报告分别给出曾成功和最终成功。不能与旧 `pd_joint_pos` 的状态 WM 数据直接混用。回放保存 `T+1` 观测、`T` 动作和奖励；官方 Buffer 的 slot 0 动作/奖励为占位 NaN，采样器会丢弃，不是真实环境的非有限值。

### 13.3 训练配方

```bash
python -m wms run tdmpc2-pushcube-train --wait
```

| 字段 | 默认值 | 说明 |
|---|---:|---|
| 顶层 `seed` | 0 | 训练随机种子 |
| `params.model_size` | 5 | 官方 MODEL_SIZE 档位，不是字符串 checkpoint 名 |
| `params.num_envs` | 32 | 并行仿真数量 |
| `params.steps` / `buffer_size` | 102400 / 102400 | 真实环境转移预算 / replay 容量 |
| `params.steps_per_update` | 8 | 每 8 条转移更新一次，须整除 num_envs |
| `params.batch_size` | 256 | 训练 batch |
| `params.checkpoint_every` | 25600 | 中间保存间隔 |
| 顶层 `timeout_s` | 14400 | 工作进程预算秒数 |
| `inputs.checkpoint` | 可选 | 已有权重 warm start 路径 |

`steps` 必须放在 `params`；顶层通用 `steps` 不控制 TD-MPC2 训练。训练按完整并行回合收集，正式 32×50 的设置建议选择整回合预算，查看报告中的实际交互数。可覆盖的规划/算法字段包括 `horizon/iterations/num_samples/num_elites/num_pi_trajs/lr` 等，实际值以 `config.resolved.json` 为准；unknown 参数不保证生效。控制器和任务在代码中受约束。

产物含 `initial.pt`、中间权重、`model.pt`、`training.json`、`training_history.jsonl`、`config.resolved.json`、`provenance.json`。checkpoint 保存模型、配置与元数据，**不含优化器、replay 和 RNG**；warm start 不是精确断点续训。

### 13.4 检查点评测与动作扰动

```python
import wms

training_id = "run_REPLACE_WITH_YOUR_TRAINING_ID"
evaluation = wms.run({
    "recipe": "tdmpc2-pushcube-eval",
    "inputs": {"source_run": training_id},
    "params": {"episodes": 10, "seed_start": 1000, "video_episodes": 2,
               "action_noise_std": 0.0, "action_scale": 1.0,
               "observation_noise_std": 0.0}
})
print(evaluation["id"])
```

把占位 ID 换成当前服务真实训练 ID。`inputs.source_run` 读取该 run 的 `model.pt`；`initial_checkpoint=true` 改读 `initial.pt`。也可使用本机可访问的 `inputs.checkpoint`，显式路径优先。`inputs.seeds=[...]` 可代替 `episodes/seed_start` 自动范围。来源 run 必须位于同一服务存储树，不能拿其他服务的同名 ID 拼接读取。

默认录像前 2 个种子、按种子顺序选择。动作干预按 `action_scale` 缩放、加 `action_noise_std` 高斯噪声，再限幅；保存 proposed/executed action。观测噪声作用于混合单位特权状态，已实现但没有正式观测噪声研究，不能称校准传感器噪声模型。

输出 `evaluation.json`、逐回合真实轨迹、MP4 和 `visualization.html`；50 步完整回合应有 51 帧（初始帧 + 每步后帧）。

### 13.5 固定动作预测

```python
prediction = wms.run({
    "recipe": "tdmpc2-pushcube-predict",
    "inputs": {"source_run": training_id,
               "evaluation_run": "run_REPLACE_WITH_COMPLETED_EVALUATION_ID"},
    "params": {"horizons": [1, 5]}
})
```

也可用显式 `checkpoint/evaluation_report` 文件路径。此配方读取真实评测轨迹，从窗口起点编码 latent，沿**实际执行动作**递归预测，然后才读取末端真实观测编码为目标；不会把未来状态反馈到预测链。

输出 `prediction_summary.json`、逐窗口 `latents_episode_XXX_hH.npz` 等，比较动作条件预测、latent persistence 和同模型零动作。reward MAE 是窗口最后一步的奖励误差，不是 H 步累计 return。latent 没有毫米等物理单位；不同 checkpoint 的表示空间不能直接用 latent MSE 排名。当前模型没有物理状态解码头。

### 13.6 完整流程客户端与通用接口

```bash
# 从头训练，再执行初始/训练后/动作噪声三组评测与预测。
python examples/tdmpc2/run_study.py --record local-study.json

# 复用同一服务的一次训练；依然会创建新评测，不会只读取旧分数。
python examples/tdmpc2/run_study.py --training-run YOUR_TRAINING_RUN_ID --record local-study-reuse.json

# 复用相同模型，以平台通用 predict 或 rollout 组件运行。
python examples/tdmpc2/use_existing_baseline.py YOUR_TRAINING_RUN_ID --task predict
python examples/tdmpc2/use_existing_baseline.py YOUR_TRAINING_RUN_ID --task rollout
```

记录文件在当前项目工作目录写入，必须遵守存储根规则。三组正式配方自动使用独立 `wms.engines.tdmpc2.metrics:measure`，需等待自己的评价任务；通用组件路径不自动附带相同研究指标。

### 13.7 既有结果与边界

既有正式训练：seed 0、102,400 次真实转移、12,800 次更新、2,048 回合；模型参数 4,972,654，记录的训练循环约 691 秒。三组共享评测 seeds 1000—1009：

| 条件 | 平均 return | 曾成功 / 最终成功 |
|---|---:|---:|
| 初始权重 | 0.747131 | 0/10 / 0/10 |
| 训练后权重 | 40.453225 | 10/10 / 10/10 |
| 训练后 + 动作噪声 std=0.1 | 38.068812 | 10/10 / 10/10 |

动作条件 latent MSE 为 H1 0.000240815、H5 0.001306656，同模型零动作分别 0.002252745/0.006119405，persistence 分别 0.002292892/0.006278533。三组共六段实际仿真视频已解码，独立评分完成；重复评价未改变被核验的来源工件。

这些结果是一项单训练 seed、有限场景基线。没有测量碰撞、接触力或风险成本，不能把 10/10 成功解释成 100% 安全；也没有完成多训练 seed、完整论文预算、跨任务或跨机器泛化。

## 14. 可选教学模板

模板不会随默认 `native/*` 注册；启动开发服务时增加相应 `--registry`：

```bash
python -m wms serve --registry templates/model/registry.json --registry templates/environment/registry.json
```

| 配方 | 能力与参数 | 验证范围 |
|---|---|---|
| `onboarding-model-predict` | 一维解析状态模型；默认 gain 1；三步动作；独立位置误差 | 已执行教学预测、HTTP/CLI/Python |
| `onboarding-model-train` | `samples=64`、`synthetic_gain=0.8`，最小二乘拟合 gain 并保存 checkpoint | 已执行教学训练及回载，不是神经 WM |
| `onboarding-model-upstream` | `UpstreamWorldModel` 骨架，需实现加载/预测及机器 runtime | 未绑定真实模型，不能当已接入结果 |
| `template-pendulum-rollout` | Gymnasium Pendulum，gravity 10、max_episode_steps 200、seed 42、40 步、恒定 0.5 N·m | 真实 CPU 仿真和独立评价 |
| `template-pendulum-video` | 同样控制协议，原生包装登记 `simulation.mp4` | 真实录像和媒体读取 |

模型模板仅需平台依赖；Pendulum 需要独立安装模板所列的 Gymnasium/渲染/编码依赖，runtime 指向其解释器。安装与替换指南分别见 [模型模板](../templates/model/README.zh-CN.md) 和 [环境模板](../templates/environment/README.zh-CN.md)。一维教学模型不能直接预测三维 Pendulum 观测，不应只为拼接成功强行转换维度。

## 15. 实际记录索引与常见问题

### 15.1 既有部署的可核对 ID

以下是截至核对日可只读查询的记录标识，不是公共下载链接，也不随 GitHub 仓库分发。新部署没有这些 ID，应生成自己的记录。

| 范围 | 已完成 source run |
|---|---|
| Wan 短 smoke / ModelScope 短 smoke | `run_0caf32854e5c4abe` / `run_bb8f96d4c5ee43fa` |
| Dreamer 后验 / 未来预测 | `run_46220d8223ae45c6` / `run_82fc6be49a7d48e3` |
| RLVR language / video | `run_d428ef1a317a4be0` / `run_88361fd2c26f4de7` |
| RENEW 短运行 / OpenVLA 修复后推理 | `run_5c554488cff54bcc` / `run_b2018c9a1d4c4cde` |
| ManiSkill 规划器 / hold | `run_9011dd499b2647ba` / `run_2d4863905c3d45a4` |
| 状态 WM 完整实验 / 录像实验 | `run_b474c67a396441c2` / `run_d57f3484d13949ec` |
| TD-MPC2 训练 / 训练后评测 | `run_d5fc89b69fb84542` / `run_c1bbe3ad8f214a8e` |
| TD-MPC2 初始 / 干预 / 预测 | `run_9eefa45505254a41` / `run_9ea6ef48e91f4787` / `run_032261dca3894c71` |

### 15.2 排障与解释

| 现象 | 应检查什么 |
|---|---|
| 配方存在但 `configured=false` | 创建对应机器 runtime，检查映射键和相对路径；模板不等于已安装资源 |
| `configured=true` 后找不到包/权重 | 查看 run 的解析后 `runtime.python/config_file`；就绪标记不做 GPU/文件预检 |
| Windows 客户端提交后 Linux 找不到 K 盘文件 | 输入数据与 runtime 路径必须在服务机器可见；客户端本地路径不会自动上传 |
| 运行很快但没有训练/视频 | 查配方任务、参数作用域、真实输出；toy 只是接口演示；各引擎不统一消费顶层 `steps` |
| 调整参数似乎无效 | 按每节的 inputs/params 位置填写；RENEW 的结构协议、Wan prompt_extend 等字段当前不是可调功能 |
| GPU 可见但仿真失败 | 区分 CUDA、Vulkan 渲染和 PhysX GPU；查看 ICD、资产与项目内 GPU 库路径 |
| 模型 completed 但评分不见了 | 检查是否 `metrics=[]`，再查看独立评价任务及 `metric_errors`；不要将原报告字段冒称自动评价 |
| 语言任务 completed 却没有有效状态 | 查看 `diagnostics.parse_status` 和 raw output；生成进程完成不等于解析成功 |
| 修改源码/配置后菜单不更新 | 新配方需服务重新加载；每次显式运行参数可直接覆盖；原生 worker 会加载当前代码 |
| “预测更准确”能否说明“更安全” | 需要预先定义危险事件、概率/代价、执行规则与真实后果对照；本仓库上述默认结果不能自动推出 |

保留解析后的配置、依赖版本、来源 commit/revision、实际输入、原始输出、失败记录与独立评价。新增实验应写新输出目录；不覆盖旧证据，也不把新协议结果合并进历史参数下的结论。
