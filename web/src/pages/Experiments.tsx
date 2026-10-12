import { useState } from 'react';
import { Config, Recipe, directionTitle, taskTitle } from '../api';
import { Empty } from '../components/Common';

interface Props {
  recipes: Recipe[]; recipeId: string; raw: string; config: Config | null; busy: boolean;
  onRecipe: (id: string) => void; onChange: (raw: string) => void; onRun: () => void;
}
export default function Experiments({ recipes, recipeId, raw, config, busy, onRecipe, onChange, onRun }: Props) {
  const [direction, setDirection] = useState('all');
  const directions = [...new Set(recipes.map(recipe => recipe.direction).filter(Boolean))];
  const visible = recipes.filter(recipe => direction === 'all' || recipe.direction === direction);
  const selected = recipes.find(recipe => recipe.id === recipeId);
  const example = /toy|demo|示例|演示/i.test([recipeId, selected?.name, ...(Array.isArray(config?.tags) ? config.tags : [])].join(' '));
  const update = (key: string, value: unknown) => config && onChange(JSON.stringify({ ...config, [key]: value }, null, 2));
  return <>
    <div className="page-heading"><div><p className="eyebrow">EXPERIMENTS</p><h1>配置一次实验</h1><p>选取配方，调整输入，在指定 Python 环境中运行。</p></div><button className="primary" disabled={busy || !config} onClick={onRun}>{busy ? '正在提交…' : '运行实验 →'}</button></div>
    <div className="builder-grid">
      <section className="panel"><div className="panel-title"><h2>实验配方</h2><span>{recipes.length} 个配方 · 配置状态不代表验证通过</span></div>
        <div className="tabs"><button className={direction === 'all' ? 'active' : ''} onClick={() => setDirection('all')}>全部</button>{directions.map(item => <button key={item} className={direction === item ? 'active' : ''} onClick={() => setDirection(item)}>{directionTitle[item] ?? item}</button>)}</div>
        <div className="recipe-list">{visible.map(recipe => <button key={recipe.id} disabled={busy} className={'recipe-card ' + (recipe.id === recipeId ? 'selected' : '')} onClick={() => onRecipe(recipe.id)}><strong>{recipe.name}</strong><span>{directionTitle[recipe.direction] ?? recipe.direction} · {taskTitle[recipe.task] ?? recipe.task}{recipe.configured === false && ' · 未配置'}</span>{recipe.description && <p>{recipe.description}</p>}{recipe.configured === false && <p className="configuration-error">{recipe.configuration_error ?? '缺少机器运行配置，可在右侧补充。'}</p>}<code>{recipe.id}</code></button>)}</div>
        {!visible.length && <Empty>暂时没有该方向的配方。可直接在右侧填写实验配置。</Empty>}
      </section>
      <section className="panel"><div className="panel-title"><h2>运行参数</h2><span>{config ? taskTitle[config.task] ?? config.task : '编辑 JSON 配置'}</span></div>
        {example && <div className="notice">当前选择的是流程示例 / toy 组件，用于检查调用与输出，不代表真实模型实验。</div>}
        {selected?.configured === false && <div className="notice warning">默认机器配置尚未就绪：{selected.configuration_error ?? '未配置'}。仍可在 JSON 中指定有效的 runtime 后提交；页面状态反映服务启动时加载的默认配方。</div>}
        <label className="field">实验名称<input value={config?.name ?? ''} disabled={!config} onChange={event => update('name', event.target.value)} placeholder="本次实验名称" /></label>
        <div className="field-grid"><label className="field">随机种子 seed<input type="number" step="1" value={config?.seed ?? 0} disabled={!config} onChange={event => update('seed', Number(event.target.value))} /></label><label className="field">交互 / 预测步数 steps<input type="number" min="1" step="1" value={config?.steps ?? 1} disabled={!config} onChange={event => update('steps', Number(event.target.value))} /></label></div>
        <label className="field">执行超时 timeout_s（秒，留空不限制）<input type="number" min="0.1" step="any" value={typeof config?.timeout_s === 'number' ? config.timeout_s : ''} disabled={!config} onChange={event => { if (!config) return; const next = { ...config }; if (event.target.value === '') delete next.timeout_s; else next.timeout_s = Number(event.target.value); onChange(JSON.stringify(next, null, 2)); }} placeholder="例如 600；不含排队时间" /></label>
        <p className="muted">在下方 JSON 中编辑 inputs、params、组件和 runtime。metrics 非空时，模型完成后自动创建独立评价任务并加入队尾。</p>
        <label className="field json-label">完整实验配置<textarea className="config-editor" spellCheck={false} value={raw} onChange={event => onChange(event.target.value)} aria-label="完整实验 JSON 配置" /></label>
        {raw && !config && <p className="text-danger">请将配置改为有效的 JSON 对象后提交。</p>}
      </section>
    </div>
  </>;
}

