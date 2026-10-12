import { useState } from 'react';
import { Run, dateText, taskTitle, terminal, valueText, runStatus, sourceRunId } from '../api';
import { Empty, JsonView, Status } from '../components/Common';

interface Props {
  runs: Run[]; run: Run | null; logs: string; busy: boolean;
  onSelect: (id: string) => void; onRefresh: () => void; onCancel: () => void;
  onRetry: () => void; onResults: () => void; onReuse: () => void;
}
export default function Runs({ runs, run, logs, busy, onSelect, onRefresh, onCancel, onRetry, onResults, onReuse }: Props) {
  const [filter, setFilter] = useState('all');
  const visible = runs.filter(item => filter === 'all' || (filter === 'evaluate' ? item.task === 'evaluate' : item.task !== 'evaluate'));
  const source = run ? sourceRunId(run) : undefined;
  return <>
    <div className="page-heading"><div><p className="eyebrow">RUNS</p><h1>运行与日志</h1><p>模型与评价分别排队、运行和记录状态；评价失败不会改变模型结果。</p></div><button className="secondary" disabled={busy} onClick={onRefresh}>刷新运行列表</button></div>
    <div className="run-layout"><section className="panel run-history"><h2>最近运行</h2><div className="tabs" aria-label="运行类型筛选">{[['all', '全部'], ['experiment', '实验'], ['evaluate', '评价']].map(([id, title]) => <button key={id} className={filter === id ? 'active' : ''} onClick={() => setFilter(id)}>{title}</button>)}</div>{visible.length ? visible.map(item => <button key={item.id} className={'run-item ' + (run?.id === item.id ? 'selected' : '')} onClick={() => onSelect(item.id)}><div><strong>{item.name || item.id}</strong><Status state={runStatus(item)} /></div><span>{taskTitle[item.task] ?? item.task} · {dateText(item.created_at)}</span><code>{item.id}</code></button>) : <Empty>该筛选下暂无运行。</Empty>}</section>
      <div>{run ? <><section className="panel"><div className="panel-title"><h2>{run.name || run.id}</h2><Status state={runStatus(run)} /></div><div className="metadata"><span>任务 <strong>{taskTitle[run.task] ?? run.task}</strong></span><span>创建 <strong>{dateText(run.created_at)}</strong></span><span>运行 ID <code>{run.id}</code></span>{source && <span>来源实验 <button className="text-button" onClick={() => onSelect(source)}>{source} →</button></span>}{run.automatic_evaluation_id && <span>自动评价 <button className="text-button" onClick={() => onSelect(run.automatic_evaluation_id!)}>{run.automatic_evaluation_id} →</button></span>}</div>{run.error != null && <div className="notice failure">{run.error_code && <strong>{run.error_code} · </strong>}{valueText(run.error)}</div>}{runStatus(run) === 'partial_failed' && <div className="notice failure">评价进程已完成，但部分指标计算失败。请查看结果中的错误；失败项不会从旧评价补值。</div>}<div className="button-row"><button className="primary" onClick={onResults}>查看结果 →</button>{run.task !== 'evaluate' && <button className="secondary" disabled={busy} onClick={onReuse}>复制配置</button>}{terminal(run.state) ? <button className="secondary" disabled={busy} onClick={onRetry}>重新运行</button> : <button className="danger-button" disabled={busy} onClick={onCancel}>停止运行</button>}</div></section>
      <section className="panel"><div className="panel-title"><h2>工作进程日志</h2><span>{terminal(run.state) ? '运行已结束' : '自动更新'}</span></div><pre className="log-view" aria-live="off">{logs || '暂无日志。'}</pre></section><details className="panel"><summary>本次运行配置</summary><JsonView value={run.config} /></details></> : <section className="panel"><Empty>选择左侧运行，查看状态、日志和配置。</Empty></section>}</div>
    </div>
  </>;
}

