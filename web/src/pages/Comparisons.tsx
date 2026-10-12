import { useEffect, useState } from 'react';
import { api, Comparison, Run, dateText, evaluationKind, runStatus, statusTitle, taskTitle, valueText } from '../api';
import { Empty, JsonView, Status } from '../components/Common';

interface Props {
  runs: Run[]; comparison: Comparison | null; busy: boolean;
  onCompare: (ids: string[], evaluations: Record<string, string>) => void; onRun: (id: string) => void;
}

export default function Comparisons({ runs, comparison, busy, onCompare, onRun }: Props) {
  const [selected, setSelected] = useState<string[]>([]);
  const [choices, setChoices] = useState<Record<string, string>>({});
  const [histories, setHistories] = useState<Record<string, Run[]>>({});
  const [historyError, setHistoryError] = useState('');
  const [changed, setChanged] = useState(false);
  const experiments = runs.filter(run => run.task !== 'evaluate');
  const metricNames = [...new Set((comparison?.rows ?? []).flatMap(row => Object.keys(row.metrics ?? {})))];
  useEffect(() => {
    if (!selected.length) return;
    let active = true;
    const load = async () => {
      try {
        const records = await Promise.all(selected.map(async id => [id, (await api<{ items: Run[] }>('/runs/' + encodeURIComponent(id) + '/evaluations')).items] as const));
        if (active) { setHistories(Object.fromEntries(records)); setHistoryError(''); }
      } catch (error) { if (active) setHistoryError(error instanceof Error ? error.message : String(error)); }
    };
    void load();
    const timer = setInterval(() => { void load(); }, 3000);
    return () => { active = false; clearInterval(timer); };
  }, [selected]);
  useEffect(() => { setChanged(false); }, [comparison]);
  const compare = () => onCompare(selected, Object.fromEntries(selected.filter(id => choices[id]).map(id => [id, choices[id]])));
  const toggle = (id: string, checked: boolean) => {
    setSelected(previous => checked ? [...previous, id] : previous.filter(item => item !== id));
    setChanged(true);
  };
  const selectEvaluation = (id: string, value: string) => {
    setChoices(previous => ({ ...previous, [id]: value }));
    setChanged(true);
  };
  return <>
    <div className="page-heading"><div><p className="eyebrow">COMPARISONS</p><h1>比较实验结果</h1><p>默认选取最近完成的独立评价；查看时间和参数后再解释差异，不合成安全总分。</p></div><button className="primary" disabled={busy || !selected.length} onClick={compare}>比较所选运行 ({selected.length})</button></div>
    <section className="panel"><h2>选择实验及评价来源</h2>{historyError && <div className="notice failure">评价历史读取失败：{historyError}</div>}{experiments.length ? <div className="table-scroll"><table><thead><tr><th>选择</th><th>实验</th><th>任务 / 状态</th><th>评价来源</th><th>创建时间</th></tr></thead><tbody>{experiments.map(run => <tr key={run.id}><td><input type="checkbox" aria-label={'选择 ' + (run.name || run.id)} checked={selected.includes(run.id)} onChange={event => toggle(run.id, event.target.checked)} /></td><td><strong>{run.name}</strong><code className="block">{run.id}</code></td><td>{taskTitle[run.task] ?? run.task}<br /><Status state={runStatus(run)} /></td><td><select className="evaluation-select" aria-label={'评价来源 ' + run.id} disabled={!selected.includes(run.id)} value={choices[run.id] ?? ''} onChange={event => selectEvaluation(run.id, event.target.value)}><option value="">自动：最近完成评价</option>{(histories[run.id] ?? []).filter(item => item.state === 'completed').map(item => <option key={item.id} value={item.id}>{dateText(item.created_at)} · {statusTitle[runStatus(item)] ?? runStatus(item)} · {item.id}</option>)}{(run.record_version ?? 0) < 3 ? <option value="legacy">旧版评价（显式选择）</option> : <option value="native">模型原生指标（显式选择）</option>}</select></td><td>{dateText(run.created_at)}</td></tr>)}</tbody></table></div> : <Empty>完成一次实验后即可在这里选择运行。</Empty>}</section>
    {comparison && <section className="panel"><div className="panel-title"><h2>指标对照</h2><span>{comparison.rows.length} 次运行 · 点击比较刷新快照</span></div>{changed && <div className="notice">选择已调整，以下仍是上次比较快照。请点击“比较所选运行”更新。</div>}{comparison.warnings?.map((warning, index) => <div key={index} className="notice warning">{warning}</div>)}
      <div className="table-scroll"><table><thead><tr><th>实验</th><th>评价来源 / 最新尝试</th>{metricNames.map(name => <th key={name}>{name}</th>)}<th>评价错误与提示</th></tr></thead><tbody>{comparison.rows.map(row => <tr key={row.run_id}><td><strong>{row.name}</strong><code className="block">{row.run_id}</code><Status state={row.state} /></td><td className="evaluation-cell"><strong>{evaluationKind[row.evaluation?.kind ?? 'none']}</strong>{row.evaluation?.id && (row.evaluation.kind === "independent" ? <button className="text-button block" onClick={() => row.evaluation?.id && onRun(row.evaluation.id)}>{row.evaluation.id} →</button> : <code className="block">{row.evaluation.id}</code>)}{row.evaluation?.created_at && <span className="block">{dateText(row.evaluation.created_at)}</span>}{row.latest_evaluation && <div className="latest-attempt"><span>最新尝试 </span><Status state={row.latest_evaluation.state === 'completed' && row.latest_evaluation.evaluation_status === 'partial_failed' ? 'partial_failed' : row.latest_evaluation.state} /><button className="text-button block" onClick={() => onRun(row.latest_evaluation!.id)}>{row.latest_evaluation.id} →</button><span className="block">{dateText(row.latest_evaluation.created_at)}</span>{row.latest_evaluation.id !== row.evaluation?.id && <p className="text-danger">此最新尝试不是当前展示的评分。</p>}{row.latest_evaluation.error != null && <p className="text-danger">{valueText(row.latest_evaluation.error)}</p>}</div>}<details className="evaluation-config"><summary>指标入口 / 参数 / 版本</summary><JsonView value={row.evaluation?.metric_specs ?? []} /></details></td>{metricNames.map(name => <td key={name}>{valueText(row.metrics?.[name])}</td>)}<td>{Object.entries(row.metric_errors ?? {}).map(([name, message]) => <p className="text-danger" key={name}><code>{name}</code>：{message}</p>)}{row.warnings?.map((warning, index) => <p className="text-danger" key={index}>{warning}</p>)}{!Object.keys(row.metric_errors ?? {}).length && !row.warnings?.length && '—'}</td></tr>)}</tbody></table></div>{!metricNames.length && <p className="muted">没有可比较的指标。失败项保持缺失；可在结果页查看评价记录及原始输出。</p>}
    </section>}
  </>;
}
