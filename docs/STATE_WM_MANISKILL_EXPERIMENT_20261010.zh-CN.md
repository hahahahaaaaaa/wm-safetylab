# 首个小型状态 WM＋ManiSkill 后果对照实验

发布说明：本文的 `K:/world-model` 与 `/srv/world-model` 均为目录示例，需按机器调整。`results/`、`data-linux/`、`source-audit/` 和历史部署文档属于本地记录，未随仓库发布；文中的运行 ID、验收数值描述当时实测，不能据此假定新部署已有这些工件。

日期：2026-10-10。当前证据状态：真实服务器预跑、正式平台 native 任务、自动及手动独立 evaluate 任务均已完成；正式任务工件和关联审计通过。

2026-10-11 补充：同配置重新采集并生成仿真录像、预测对照视频和逐帧数据，见 [录像与同步对照记录](STATE_WM_MANISKILL_VIDEO_20261011.zh-CN.md)。本文的原运行未录制视频，保持原样；新运行单独保存。

## 完成了什么

在已有 ManiSkill 3.0.1 环境中，实际运行 Panda 的 PushCube-v1，采集 36 个 episode / 2217 条状态转移。固定官方运动规划器，对七个机械臂关节的绝对目标分别添加标准差 0.003 rad 的高斯扰动，保留扰动前计划动作和实际限幅执行动作。模型使用实际执行动作。

训练了一个真正的小型神经动作条件状态世界模型：两层 128 隐单元的 SiLU MLP，预测下一状态的增量；另训练同结构无动作 MLP 作为对照。每种模型使用训练 seed 0、1、2，各最多 400 epochs，验证集选择 checkpoint。两个模型参数量分别为 23,832 和 22,808；不是严格等参数模型。

采集和 6 次 CPU 训练的预跑计时约 27.84 秒，正式平台全新采集与训练计时约 28.55 秒。此数字只描述当前小模型和当前服务器这一次运行，不作为其它模型或机械臂任务的工期/算力保证。没有安装依赖、修改现有原生环境、权重或旧结果，没有停止其他 GPU 服务。

## 四层的位置

| 层 | 本实验实际实现 |
|---|---|
| 数据层 S1 | 仿真采集、计划/实际动作、episode 与随机种子、真实后果、固定分割 |
| 模型层 S2/S3 | 24 维特权状态；有动作/无动作残差 MLP；一步与五步递归想象；checkpoint |
| 评估与规划层 S4 | 固定的原生运动规划器用于采集；评价预测误差。学习到的 WM 没有参与规划决策 |
| 执行与反馈层 S5/S6 | Panda 实际仿真执行、真实后果保存、独立后果对照指标；反馈更新实验尚未做 |

## 数据和控制语义

- 环境：PushCube-v1；机器人：Panda；控制模式：pd_joint_pos；physx_cpu；控制频率 20 Hz，仿真频率 100 Hz。
- 状态：9 维 qpos + 9 维 qvel + 3 维 cube 世界坐标 + 3 维 TCP 世界坐标，共 24 维。机械臂角度/速度为 rad、rad/s，夹指位置/速度为 m、m/s，空间位置为 m。
- 动作：7 个绝对关节目标（rad）＋1 个归一化夹爪命令。
- 本表示是部分特权状态，不包含图像、完整接触状态、物体角速度或完整世界状态；不能直接用于视觉感知结论。
- 环境 seed 0—35；扰动种子为环境 seed + 1,000,000；分割 seed 20261010。
- 训练 24 episodes / 1494 transitions；验证 6 / 361；测试 6 / 362。测试 episode 为 3、4、17、18、21、35。
- 5 步预测从真实测试初始状态出发，后续状态由模型递归产生，只继续输入已经记录的实际动作。共有 338 个有效五步窗口；5 个控制步相当于 0.25 秒。
- 此次 36 个采集 episode 都曾达到任务成功，说明扰动强度下采集分布较容易，不能代表高风险任务覆盖。

## 实际结果

先对每个 episode 内各时刻误差平均，再对 episode 等权平均，最后平均 3 个训练 seed。下表数值越低越好。Persistence 是直接预测状态不变，无训练随机性。

| 对照 | 一步 cube mm | 五步 cube mm | 一步 TCP mm | 五步 TCP mm | 一步关节 RMSE rad | 五步关节 RMSE rad |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 1.7196 | 8.3913 | 5.1699 | 25.8507 | 0.007450 | 0.037060 |
| 无动作状态 MLP | 0.6458 | 2.7862 | 1.1294 | 5.9387 | 0.001578 | 0.008449 |
| 动作条件状态 MLP | 0.5310 | 1.9325 | 0.9412 | 3.6548 | 0.001150 | 0.004628 |

正式平台使用全新仿真采集，未复用预跑数据；独立评价的上述 18 个指标与预跑数值一致。这次固定数据划分支持“动作条件在这一分布上的后果预测有帮助”，也验证了真实仿真—学习—后果对照链路。不能据此推导安全措施有效、概率校准良好、跨任务泛化或论文方法优于已有顶会方法。

逐训练 seed、逐测试 episode 的结果及 seed 标准差保存在 `state_wm_metrics.json`。标准差不是置信区间，6 个测试 episode 也不足以给出广泛泛化结论。

## 新增代码和接口

```text
src/wms/engines/state_wm_maniskill/
  collect.py   真实仿真采集，输出 dataset.npz + dataset.json
  learn.py     分割、训练、验证集选模、heldout 预测与 checkpoint
  native.py    平台原生入口 run(config, output_dir, context)
  metrics.py   独立 measure(result, params)，读来源、写评价目录
native/state_wm_maniskill/recipes.json
```

配方 ID：`state-wm-maniskill`。模型任务完成后，由平台基于配方 metrics 创建独立评价。已有数据可通过 `inputs.dataset_dir` 复用，但会复制到新的模型任务目录后重新训练，以便该次任务具有完整工件。

服务器已采集数据位置：

`/srv/world-model/deployments/wm-safetylab/results/state-wm-engineering-20261010/full-1/output`

通过 Python 客户端发起示意（新版平台已验证）：

```python
import wms

run = wms.run({
    "recipe": "state-wm-maniskill",
    "inputs": {
        "dataset_dir": "/srv/world-model/deployments/wm-safetylab/results/state-wm-engineering-20261010/full-1/output"
    }
})
# 不提供 dataset_dir 就重新采集 36 episodes。
# 模型等待和评价等待分别进行，使用服务返回的任务 ID。
```

## 证据位置与检查

本地对应证据均在 `K:/world-model/deployments/wm-safetylab/results/state-wm-engineering-20261010/`：

- `full-1/output/`：数据、六个模型 checkpoint、预测、训练和运行记录、原始 result.json。
- `full-1/evaluation-preview/state_wm_metrics.json`：平台上线前的独立函数预评价，不能冒称平台 evaluate 任务。
- `audit-1/audit.json`：划分互斥且覆盖全数据、有限数值与维度、计划/实际动作不同、episode 内前后状态对齐、六个 checkpoint 可读取、指标可重算、全部来源文件哈希未改变。
- `run_standalone.py` 和 `audit_evidence.py`：可复验脚本。

## 后续研究还缺什么

本轮没有完成“概率—反馈—更新目标—固定决策”的完整科研主张：当前是确定性均值预测，不是校准的后果概率分布；没有比较更新前后，也没有让 WM 改变候选动作选择；碰撞、接触力和危险事件尚未作为本模型训练/评价目标。

下一项研究应在固定数据和独立测试协议上单独加入后果分布建模、固定决策规则和反馈更新对照。不要将本轮的工程误差改善直接命名为安全性改善。

## 正式平台任务与独立评价验收

本轮通过网页提交，服务实际运行的记录：

| 类型 | 任务 ID | 实测状态 | 来源／作用 |
|---|---|---|---|
| 模型任务 | run_b474c67a396441c2 | completed | 全新采集 36 episodes，再训练 2×3 个模型 |
| 自动评价 | run_6359aaa32c114f95 | completed，0 指标错误 | 模型完成后队尾自动入队 |
| 网页手动评价 | run_38297c3027934bf8 | completed，0 指标错误 | 重新评价同一模型来源，不覆盖自动评价 |

模型任务创建于 22:58:31，完成于 22:59:06；自动评价 22:59:18 完成；手动评价 22:59:28 完成（均为 2026-10-10，Asia/Shanghai）。API 明确记录自动评价 ID，以及两条评价的 source_run 关联。

通过实际服务器文件重新审计：数据与 episode 分割一致；6 个 checkpoint 可加载；自动／手动评价指标与重新计算的 18 个指标逐项一致；两条评价的配置和结果均指向正确来源；审计重算前后来源目录每个文件的 SHA-256 均不变。共 8 项检查通过。这里的哈希仅用于这次验收证据，不是平台兼容性规则。

正式工件在服务器 `data-linux/runs/<任务ID>/`。本地完整副本在：

`K:/world-model/deployments/wm-safetylab/results/state-wm-engineering-20261010/platform-1/`

其中三个以任务 ID 命名的目录保存各自配置、日志及 output；`api/` 保存实际 HTTP 运行记录与评价列表；`platform-audit-1/audit.json` 保存正式来源／真实独立评价核对记录。预跑 `full-1` 和 `audit-1` 原位保留，没有覆盖。

正式实验后，入口改为复用共享原生配置助手，保证显式 CUDA 配置优先；此前入口已保留为 `platform-1/native_before_runtime_env_fix.py`。最终代码在服务器另跑 6 episodes/30 epochs/1 seed：模型 `run_2dac811cac17445c`、独立评价 `run_9b77f4700d314e9d` 均完成，0 指标错误。这是最终集成检查，不替换本文 36 episodes 的正式结果，记录见 `results/core-refinement-20261010/final-smoke.json`。
