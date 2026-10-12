import { statusTitle } from '../api';

export function Status({ state }: { state: string }) {
  return <span className={'status status-' + state}>{statusTitle[state] ?? state}</span>;
}
export function Empty({ children }: { children: React.ReactNode }) {
  return <div className="empty">{children}</div>;
}
export function JsonView({ value }: { value: unknown }) {
  return <pre className="json-view">{JSON.stringify(value, null, 2)}</pre>;
}
export function ErrorPanel({ message, onClose }: { message: string; onClose: () => void }) {
  return <div className="error-panel" role="alert"><span>{message}</span><button aria-label="关闭错误提示" onClick={onClose}>×</button></div>;
}

