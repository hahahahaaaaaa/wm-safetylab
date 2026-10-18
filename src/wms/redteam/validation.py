"""Server-side validation of a red-team configuration.

The web UI shows these issues per setup step; compatibility rules live only here.
"""
import re

from wms.redteam.schema import Catalog, Estimate, ExperimentConfig, ResolvedConfig, StepIssue, ValidationResult

LOCAL_VRAM_GB = 8
URL = re.compile(r"^https?://[^\s/]+(:\d+)?(/\S*)?$")


def validate_config(config: ExperimentConfig, catalog: Catalog) -> ValidationResult:
    issues: list[StepIssue] = []
    add = lambda step, field, message, severity="error": issues.append(StepIssue(step=step, field=field, message=message, severity=severity))
    targets = {t.id: t for t in catalog.targets}
    envs = {e.id: e for e in catalog.environments}
    tasks = {t.id: t for t in catalog.tasks}
    plugins = {p.id: p for p in catalog.plugins}
    strategies = {s.id: s for s in catalog.strategies}
    defenses = {d.id for d in catalog.defenses}

    if not config.name.strip():
        add("target", "name", "请填写实验名称")
    target = targets.get(config.target.id)
    if target is None:
        add("target", "target.id", f"未知的被测目标：{config.target.id}")
    elif target.requires_endpoint and not URL.match((config.target.endpoint or "").strip()):
        add("targetcfg", "target.endpoint", "请填写有效的策略服务地址（http:// 或 https://）")
    elif target.status == "download_required":
        add("target", "target.id", f"{target.name} 首次运行需要下载权重", "warning")

    if len(config.scenario.purpose.strip()) < 10:
        add("app", "scenario.purpose", "请描述机器人的用途（至少 10 个字）")

    compatible = [e for e in (target.environments if target else []) if e in envs]
    env = config.environment if config.environment in compatible else (compatible[0] if compatible else None)
    if config.environment and config.environment not in compatible:
        add("envtask", "environment", f"{target.name if target else '该目标'} 不支持 {config.environment}，已改用 {env}", "warning")
    if env is None:
        add("envtask", "environment", "没有可用的仿真环境")

    unknown_tasks = [t for t in config.tasks if t not in tasks]
    for t in unknown_tasks:
        add("envtask", "tasks", f"未知任务：{t}")
    effective_tasks = [t for t in config.tasks if t in tasks and env in tasks[t].environments]
    skipped = [t for t in config.tasks if t in tasks and env not in tasks[t].environments]
    if skipped:
        add("envtask", "tasks", f"{len(skipped)} 个任务在当前环境中没有实现，将被跳过", "warning")
    if not effective_tasks:
        add("envtask", "tasks", "至少选择 1 个当前环境可用的任务")

    selected_plugins = [p for p in config.plugins if p.id in plugins]
    for p in config.plugins:
        if p.id not in plugins:
            add("plugins", "plugins", f"未知插件：{p.id}")
    if not selected_plugins:
        add("plugins", "plugins", "至少选择 1 个风险插件")
    covered = {p for t in effective_tasks for p in tasks[t].plugins}
    for p in selected_plugins:
        if p.id not in covered:
            add("plugins", "plugins", f"已选任务都不覆盖“{plugins[p.id].name}”，不会生成用例", "warning")

    usable, unsupported = [], []
    for s in config.strategies:
        if s == "basic":
            continue
        if s not in strategies:
            add("strategies", "strategies", f"未知策略：{s}")
        elif strategies[s].environments and env not in strategies[s].environments:
            unsupported.append(s)
        else:
            usable.append(s)
    if unsupported:
        add("strategies", "strategies", f"{', '.join(unsupported)} 在当前环境不可用，已忽略", "warning")
    if any(strategies[s].needs_attacker for s in usable) and config.attacker not in {a.id for a in catalog.attackers}:
        add("strategies", "attacker", "所选策略需要攻击模型")

    r = config.graders.rules
    for field, value in (("speed_max_mps", r.speed_max_mps), ("force_max_n", r.force_max_n), ("warn_distance_m", r.warn_distance_m)):
        if not value > 0:
            add("graders", f"graders.rules.{field}", "阈值需大于 0")
    if config.graders.semantic.enabled and config.graders.semantic.model not in {j.id for j in catalog.judges}:
        add("graders", "graders.semantic.model", "请选择语义判定模型")

    defs = [d for d in dict.fromkeys(config.defenses) if d in defenses]
    if "none" not in defs:
        add("defense", "defenses", "必须保留“无防护”作为对照")

    if config.run.location == "local":
        if env and not envs[env].local_allowed:
            add("review", "run.location", f"{envs[env].name} 只能在实验服务器运行")
        if target and (target.vram_gb or 0) > LOCAL_VRAM_GB:
            add("review", "run.location", f"本机显存 {LOCAL_VRAM_GB} GB，{target.name} 需约 {target.vram_gb:g} GB")

    strategies_effective = ["basic", *usable]
    episodes = sum(p.num_tests for p in selected_plugins) * len(strategies_effective) * max(len(defs), 1)
    parallel = config.run.max_concurrency if config.run.location == "server" else 1
    minutes = episodes * (target.seconds_per_episode if target else 0) / parallel / 60 * (0.7 if config.run.early_stop else 1)
    resolved = ResolvedConfig(environment=env, environment_locked=len(compatible) == 1, compatible_environments=compatible,
                              robot=envs[env].robot if env else None, tasks=effective_tasks, skipped_tasks=skipped,
                              strategies=strategies_effective, unsupported_strategies=unsupported, defenses=defs)
    return ValidationResult(valid=not any(i.severity == "error" for i in issues), issues=issues, resolved=resolved,
                            estimate=Estimate(episodes=episodes, minutes=round(minutes, 1)))
