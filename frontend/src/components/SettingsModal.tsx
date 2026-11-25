import { X } from "lucide-react";
import type { UsageSummary } from "../types";

interface SettingsModalProps {
  open: boolean;
  apiKey: string;
  serviceStatus: "checking" | "ready" | "offline";
  usage: UsageSummary | null;
  theme: "light" | "dark";
  onClose: () => void;
  onApiKeyChange: (value: string) => void;
  onApiKeyCommit: () => void;
  onToggleTheme: () => void;
}

export function SettingsModal({
  open,
  apiKey,
  serviceStatus,
  usage,
  theme,
  onClose,
  onApiKeyChange,
  onApiKeyCommit,
  onToggleTheme
}: SettingsModalProps) {
  if (!open) return null;

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <section
        className="dialog-modal settings-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-modal-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <div>
            <span className="eyebrow">WORKSPACE SETTINGS</span>
            <h2 id="settings-modal-title">工作台设置与用量统计</h2>
          </div>
          <button type="button" className="icon-close" onClick={onClose} aria-label="关闭">
            <X size={18} />
          </button>
        </div>

        <div className="settings-group">
          <label className="settings-field">
            <span>API Key（可选，留空表示未启用鉴权）</span>
            <input
              type="password"
              value={apiKey}
              onChange={(event) => onApiKeyChange(event.target.value)}
              onBlur={onApiKeyCommit}
              onKeyDown={(event) => {
                if (event.key === "Enter") event.currentTarget.blur();
              }}
              placeholder="输入 X-API-Key"
              autoComplete="off"
            />
          </label>

          <div className="settings-row">
            <span>后端连接状态</span>
            <div className={`service-pill ${serviceStatus}`}>
              <i />
              {serviceStatus === "ready"
                ? "服务就绪"
                : serviceStatus === "checking"
                  ? "检查中"
                  : "连接异常"}
            </div>
          </div>

          <div className="settings-row">
            <span>外观主题（TraceMind 暖米 / 深苔藓）</span>
            <button type="button" className="ghost-btn" onClick={onToggleTheme}>
              当前：{theme === "dark" ? "深色模式" : "浅色模式"}（点击切换）
            </button>
          </div>
        </div>

        {usage && (
          <div className="usage-panel">
            <span className="eyebrow">MODEL USAGE</span>
            <div className="usage-grid">
              <div>
                <strong>{usage.calls}</strong>
                <span>总调用次数</span>
              </div>
              <div>
                <strong>{usage.total_tokens.toLocaleString()}</strong>
                <span>累计 Tokens</span>
              </div>
              <div>
                <strong>${usage.estimated_cost_usd.toFixed(4)}</strong>
                <span>估算费用</span>
              </div>
            </div>
            {usage.by_model.length > 0 && (
              <div className="usage-models">
                {usage.by_model.map((item) => (
                  <div className="usage-model-row" key={item.model}>
                    <span>{item.model}</span>
                    <small>
                      {item.calls} 次 · {item.total_tokens.toLocaleString()} tok
                    </small>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
