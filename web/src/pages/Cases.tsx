import { useState } from 'react';
import { Run, Result, JsonObject, dateText, fileUrl, runStatus, sourceRunId, terminal, valueText } from '../api';
import { Empty, JsonView, Status } from '../components/Common';

function Metrics({ metrics, errors = {} }: { metrics: JsonObject; errors?: Record<string, string> }) {
  return <>{Object.keys(metrics).length ? <div className="metrics-grid">{Object.entries(metrics).map(([name, value]) => <div className="metric" key={name}><span>{name}</span><strong>{valueText(value)}</strong></div>)}</div> : <p className="muted">暂无指标值。</p>}{Object.keys(errors).length > 0 && <div className="notice failure"><strong>指标计算失败 · 不使用旧值补齐</strong>{Object.entries(errors).map(([name, message]) => <p key={name}><code>{name}</code>：{message}</p>)}</div>}</>;
}

function Outputs({ runId, result }: { runId: string; result: Result }) {
  return <section className="panel"><div className="panel-title"><h2>输出文件</h2><span>{result.outputs?.length ?? 0} 个文件</span></div><div className="output-grid">{(result.outputs ?? []).map((output, index) => {
    const url = fileUrl(runId, output.path);
    const isVideo = output.media_type?.startsWith('video/') || /\.(mp4|webm|ogg)$/i.test(output.path);
    const isImage = output.media_type?.startsWith('image/') || /\.(png|jpe?g|gif|webp)$/i.test(output.path);
    return <article className="output-card" key={index}><div><strong>{output.kind}</strong><a href={url} target="_blank" rel="noreferrer">打开 / 下载 ↗</a></div><code>{output.path}</code>{isVideo ? <video src={url} controls preload="metadata" /> : isImage ? <img src={url} alt={output.path} loading="lazy" /> : <p className="muted">文件保留原始格式，可下载分析。</p>}</article>;
  })}</div>{!result.outputs?.length && <p className="muted">本次任务未返回输出文件。</p>}</section>;
}

interface Props {
  run: Run | null; result: Result | null; evaluations: Run[]; busy: boolean;
  onEvaluate: () => void; onRun: (id: string) => void; onSource: (id: string) => void;
}

export default function Cases({ run, result, evaluations, busy, onEvaluate, onRun, onSource }: Props) {
  const [selected, setSelected] = useState('');
  const isEvaluation = run?.task === 'evaluate';
  const latest = evaluations[0];
  const evaluation = selected ? evaluations.find(item => item.id === selected) : latest;
  const source = run ? sourceRunId(run) : undefined;
  const metricErrors = result?.metric_errors ?? run?.metric_errors ?? {};
  const legacy = (run?.record_version ?? 0) < 3;
  const score = () => { setSelected(''); onEvaluate(); };
  return <>
    <div className="page-heading"><div><p className="eyebrow">RESULTS</p><h1>{isEvaluation ? '评价结果' : '实验结果'}</h1><p>原始输出与独立评价分别保存，每次评分都有自己的记录、日志和参数。</p></div>{!isEvaluation && <button className="secondary" disabled={busy || !run || run.state !== 'completed' || !run.config.metrics?.length} onClick={score}>{busy ? '正在提交…' : '新建独立评价'}</button>}</div>
    {!run ? <section className="panel"><Empty>请先在运行页选择一次实验。</Empty></section> : <>
      <div className="context-line"><strong>{run.name}</strong><Status state={runStatus(run)} /><code>{run.id}</code><button className="text-button" onClick={() => onRun(run.id)}>日志 / 运行操作 →</button>{source && <button className="text-button" onClick={() => onSource(source)}>来源实验 →</button>}</div>
      {run.error != null && <div className="notice failure">{valueText(run.error)}</div>}
      {!isEvaluation && <section className="panel"><div className="panel-title"><h2>独立评价记录</h2><span>{evaluations.length} 次 · 自动刷新</span></div>
        {!evaluations.length ? <p className="muted">暂无独立评价。配置中的 metrics 非空时，模型完成后自动加入评价队列；也可手动新建评分任务。</p> : <>
          <label className="field">查看评价<select value={selected} onChange={event => setSelected(event.target.value)}><option value="">最新评价尝试 · {latest?.id}</option>{evaluations.map(item => <option key={item.id} value={item.id}>{dateText(item.created_at)} · {item.id} · {runStatus(item)}</option>)}</select></label>
          {latest && evaluation?.id !== latest.id && <div className="notice">当前查看历史评价。最新尝试 <code>{latest.id}</code> <Status state={runStatus(latest)} />，创建于 {dateText(latest.created_at)}。</div>}
          {evaluation && <><div className="context-line"><Status state={runStatus(evaluation)} /><code>{evaluation.id}</code><span>{dateText(evaluation.created_at)}</span><button className="text-button" onClick={() => onRun(evaluation.id)}>日志 / 取消 / 重试 →</button><button className="text-button" onClick={() => onSource(evaluation.id)}>全部评价产物 →</button></div>
            {!terminal(evaluation.state) && <div className="notice">评价{evaluation.state === 'queued' ? '正在 FIFO 队列中等待' : '正在执行'}。模型运行状态与评分状态相互独立。</div>}
            {evaluation.error != null && <div className="notice failure">{valueText(evaluation.error)}</div>}
            {evaluation.result && <Metrics metrics={evaluation.result.metrics ?? {}} errors={evaluation.result.metric_errors ?? evaluation.metric_errors} />}
            <details className="evaluation-config"><summary>本次指标入口、参数和版本</summary><JsonView value={evaluation.result?.metric_specs ?? evaluation.config.metrics ?? []} /></details>
          </>}
        </>}
      </section>}
      {!result ? <section className="panel"><Empty>本次运行暂时没有结果。可在运行页查看日志。</Empty></section> : <>
        <section className="panel"><div className="panel-title"><h2>{isEvaluation ? '本次独立评价指标' : legacy ? '旧版评价指标（只读）' : '模型原生指标'}</h2><span>{isEvaluation ? '只展示本次计算结果' : '原始记录 · 不受重评分改写'}</span></div>
          {!isEvaluation && <p className="muted source-metric-note">{legacy ? '历史指标保留原值，评价版本可能未知；未与独立评价合并。' : '这里展示模型执行时返回的原生指标，未与后续独立评价合并。'}</p>}
          <Metrics metrics={result.metrics ?? {}} errors={metricErrors} />
          {isEvaluation && <details className="evaluation-config"><summary>本次指标入口、参数和版本</summary><JsonView value={result.metric_specs ?? run.config.metrics ?? []} /></details>}
        </section>
        <Outputs runId={run.id} result={result} />
        <details className="panel"><summary>完整结果 JSON</summary><JsonView value={result} /></details>
      </>}
    </>}
  </>;
}
