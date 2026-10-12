# 接入模型和仿真环境：开发总指南

版本：当前 WM-SafetyLab 0.3.0；更新：2026-10-11。本文指向实际可运行的模板，不使用历史 v1/v2 接口。

## 先读哪几个文件

1. 模型开发者先读 [模型教程](../templates/model/README.zh-CN.md)，然后看 `adapter.py`、`registry.json`、`metrics.py`。要接真实论文实现时再改 `upstream.py`，训练入口参考 `train.py`。
2. 环境开发者先读 [环境教程](../templates/environment/README.zh-CN.md)，然后看 `lab_env/adapter.py`、`registry.json`、`policy.py`。录像/多回合流程参考 `native.py`。
3. 需要看平台如何调用组件时，只读 `src/wms/runners/unified.py` 与 `src/wms/core/protocols.py`。一般不必修改它们。

## 代码位置与共同边界

```text
wm-safetylab/
├── templates/
│   ├── model/                 # 模型成员独立开发入口
│   │   ├── wms_model_template/
│   │   │   ├── adapter.py     # 已运行教学预测器
│   │   │   ├── upstream.py    # 真实模型加载/推理骨架
│   │   │   ├── train.py       # 原生训练包装
│   │   │   └── metrics.py     # 独立评价
│   │   └── registry.json
│   ├── environment/           # 环境成员独立开发入口
│   │   ├── lab_env/
│   │   │   ├── adapter.py     # 已运行真实 Pendulum
│   │   │   ├── policy.py      # 教学恒定力矩执行器
│   │   │   ├── native.py      # 录像循环/文件登记
│   │   │   └── metrics.py
│   │   └── registry.json
│   └── verify_onboarding.py   # 独立服务端到端复验
└── src/wms/                   # 现有共享框架，本次未修改
```

这两个目录是起点，不必永久用模板名称。正式接入可将模型代码放在 `src/wms/engines/<模型名>/`、环境代码放在 `src/wms/scenarios/<环境名>/`；也可以继续作为外部包，通过 `python_paths` 引入。共享配方放在 `native/<组件名>/recipes.json` 或独立注册表文件。依赖、权重和机器专属路径放在各自运行配置中，上游源代码保持干净。

## 最小接口对照

| 对象 | 当前调用 | 返回什么 |
|---|---|---|
| 状态/动作条件预测模型 | `Model(config).predict(request)` | `{"predictions": ...}`，或当前任务支持的文件引用 |
| 视频/语言生成模型 | `Model(config).generate(request)` | 实际生成文件的 `outputs` 与必要元数据 |
| 直接动作策略 | `Model(config).act(observation, instruction)` | 仿真环境接受的动作 |
| 环境初始化 | `Env(config).reset(seed, inputs)` | 初始观测本身 |
| 环境步进 | `Env.step(action)` | `obs/reward/terminated/truncated`，可选真实的 `cost/events/info` |
| 原生训练/复杂循环 | `run(config, output_dir, context)` | checkpoint、轨迹、视频、训练来源等工件 |
| 独立评价 | `measure(result, params)` | 指标字典；原始目录只读，评价目录写新文件 |

`config` 是组件初始化参数，`request` 是本次预测/生成输入，二者不要混用。worker 的预测请求由 `inputs`、`params`、`seed`、`output_dir` 合成，重名时 `params` 覆盖 `inputs`。模型只需要实现被选中任务使用的方法。

以下调用关系对应当前四层职责：

```text
数据层 S1：inputs / 数据文件 / Env.reset
    ↓
模型层 S2–S3：模型加载 / 记忆管理 / predict 或 generate
    ↓
算法层 S4：可选 planner / cost / 方法包装
    ↓
执行与反馈层 S5–S6：act → Env.step → 轨迹 → 独立评价
                                             ↘ native 训练/后训练 → 新 checkpoint
```

后训练不是评分后自动发生的操作；需要接入者明确实现训练目标、数据选择和更新函数。一个实验也不必使用所有层，例如离线视频生成可以完全不创建环境。

## 两人如何并行开发

| 负责人 | 负责内容 | 给对方的最小交付 |
|---|---|---|
| 模型接入 | 官方推理跑通、权重加载、输入输出转换、训练包装 | 一个可调用的模型类；一个实际输入/输出样本；依赖和权重说明 |
| 仿真接入 | 场景初始化、动作执行、观测/事件导出、录像 | 一个可调用的环境类；一段真实轨迹；动作单位/范围和观测定义 |
| 集成负责人（可兼任） | 组合配置、评价函数、配对实验与网页检查 | 一份能复现的配方及一次端到端结果 |

在开始组合前，一起确定观测字段、动作维度和单位、坐标系、控制频率、预测时域，以及终止与截断语义。模板里的“一维位置增量模型”和“三维单摆观测/力矩环境”不能直接组合；这正是格式统一与任务语义兼容的区别。

## 一次接入按什么顺序做

1. **原生最小输入跑通**：先用官方调用确认依赖、资源和输出，保存命令与版本。
2. **复制对应模板**：改包名、入口和配方 ID，替换实际后端调用。
3. **单组件运行**：模型先做离线预测；环境先配固定教学策略，记录真实轨迹。
4. **单独评价**：指标读取来源文件，生成独立评价记录；没有真值/安全传感器时明确不可评价。
5. **组合实验**：在相同真实初始状态和明确动作语义下，对齐预测与实际后果；再增加规划或防护。
6. **登记到开发服务**：用 `--registry` 加载名称；或者先提交完整配置验证，成功后再放入共享配方。

模型与环境在一个普通配对任务内共用一个解释器。不同任务可使用不同解释器；跨环境模型服务/RPC 需要额外开发，当前模板不提供。Windows 上解析出的 K 盘路径不能直接发送给 Linux 服务使用。

## 最容易遇到的五个问题

| 现象 | 原因与处理 |
|---|---|
| `ModuleNotFoundError` | 检查实际工作进程的 Python 和 `runtime.python_paths`，不是仅检查当前终端 |
| 找不到 checkpoint | 路径必须在服务机器上可见；自定义权重字段由适配器读取，注册表不会自动传递 |
| 修改配置后下拉列表没变化 | 注册表在服务启动时加载；运行输入可直接覆盖，新名称需正常部署或完整内联提交 |
| 覆盖模型参数后入口丢失 | 原模型是字符串 ID 时，用完整的 `entrypoint + config` 对象覆盖，不能只传 `config` |
| 模型结束却没有最终评分 | 自动评价是独立排队任务，要查看它自己的状态和错误；不是向原 result.json 写分数 |

另一个配置优先级细节：机器文件默认值 < 配方值 < 本次显式运行值。若共享配方写死 `runtime.python="auto"`，会盖过机器文件指定的解释器；所以模板共享配方省略该项，由目标机器决定。

## 本轮验证及尚未验证的内容

已在 Windows Python 3.11 实际运行两套模板：教学预测、教学训练、权重回载、真实 Pendulum 交互和录像；四条自动评价完成。HTTP、公共 Python 客户端、CLI 和视频文件读取均通过。环境运行使用独立的 `K:/world-model/envs/wms-onboarding`，没有向现有 ManiSkill/Wan/OpenVLA 环境安装依赖。

验证记录：`results/onboarding-20261011/http-acceptance/summary.json`；仿真独立运行在 `env-state/`、`env-direct/`；接口约定测试记录在 `contracts-all.xml`。

真实论文模型骨架尚未绑定任何新神经模型；新机器人仿真任务、Linux 上同一锁文件安装、跨环境 RPC 都不在本轮已验证范围内。替换具体模型或仿真器后，应保留一条官方原生输出作为对照。

官方接口参考：[Gymnasium Env API](https://gymnasium.farama.org/api/env/)、[Pendulum-v1](https://gymnasium.farama.org/environments/classic_control/pendulum/)。
