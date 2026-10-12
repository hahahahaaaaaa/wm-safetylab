export type JsonObject = Record<string, unknown>;
export interface Config extends JsonObject {
  name: string; task: string; direction?: string; runtime: JsonObject | string;
  inputs?: JsonObject; params?: JsonObject; seed?: number; steps?: number;
  metrics?: unknown[]; tags?: string[];
}
export interface Recipe { id: string; name: string; task: string; direction: string; description?: string; configured?: boolean; configuration_error?: string }
export interface Output { kind: string; path: string; media_type?: string }
export interface Result { status: string; outputs: Output[]; metrics: JsonObject; data?: unknown; error?: unknown; metric_errors?: Record<string, string>; metric_specs?: unknown[]; evaluation_status?: string }
export interface Run {
  id: string; name: string; task: string; direction?: string; state: string;
  created_at: string; updated_at?: string; config: Config; result: Result | null; error?: unknown;
  metric_errors?: Record<string, string>;
  source_run?: string; automatic_evaluation_id?: string; evaluation_status?: string; error_code?: string; record_version?: number;
}
export interface EvaluationReference { id?: string; kind: 'independent' | 'legacy' | 'native' | 'none'; created_at?: string; state?: string; metric_specs?: unknown[] }
export interface EvaluationAttempt { id: string; state: string; created_at: string; error?: unknown; evaluation_status?: string }
export interface ComparisonRow {
  run_id: string; name: string; state: string; task?: string; metrics: JsonObject; metric_errors?: Record<string, string>;
  evaluation?: EvaluationReference; latest_evaluation?: EvaluationAttempt | null; warnings?: string[];
}
export interface Comparison { rows: ComparisonRow[]; warnings?: string[] }
export interface LogChunk { text: string; next_offset: number }

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch('/api/v1' + path, {
    ...init, headers: { 'Content-Type': 'application/json', ...init?.headers },
  });
  const text = await response.text();
  let data: unknown;
  try { data = text ? JSON.parse(text) : null; }
  catch { throw new Error('服务返回了非 JSON 响应，请检查后台连接。'); }
  if (!response.ok) {
    const error = data as { message?: string; detail?: unknown } | null;
    throw new Error(error?.message ?? valueText(error?.detail ?? data));
  }
  return data as T;
}
export const post = <T,>(path: string, value: unknown = {}) => api<T>(path, { method: 'POST', body: JSON.stringify(value) });
export const terminal = (state: string) => ['completed', 'failed', 'cancelled', 'interrupted'].includes(state);
export const statusTitle: Record<string, string> = { queued: '排队中', running: '运行中', cancelling: '正在取消', completed: '已完成', partial_failed: '部分指标失败', failed: '失败', cancelled: '已取消', interrupted: '已中断' };
export const directionTitle: Record<string, string> = { control: '具身控制', video: '视频', vla: 'VLA / 行动', driving: '自动驾驶', demo: '流程示例' };
export const taskTitle: Record<string, string> = { generate: '生成', predict: '预测', rollout: '环境交互', plan: '规划', as_env: '模型环境', native: '原生任务', train: '训练', evaluate: '独立评价' };
export const runStatus = (run: Run) => run.state === 'completed' && (run.evaluation_status === 'partial_failed' || run.result?.evaluation_status === 'partial_failed' || (run.task === 'evaluate' && Object.keys(run.result?.metric_errors ?? run.metric_errors ?? {}).length > 0)) ? 'partial_failed' : run.state;
export const sourceRunId = (run: Run) => run.source_run ?? (typeof run.config.source_run === 'string' ? run.config.source_run : undefined);
export const evaluationKind: Record<string, string> = { independent: '独立评价', legacy: '旧版评价（只读）', native: '模型原生指标', none: '无可用评价' };
export const valueText = (value: unknown): string => value === null || value === undefined ? '—' : typeof value === 'number' ? Number(value.toFixed(6)).toString() : typeof value === 'string' ? value : JSON.stringify(value);
export const dateText = (value: string) => new Date(value).toLocaleString('zh-CN', { hour12: false });
export const fileUrl = (runId: string, path: string) => '/api/v1/runs/' + encodeURIComponent(runId) + '/files/' + path.split(/[\\/]/).map(encodeURIComponent).join('/');

