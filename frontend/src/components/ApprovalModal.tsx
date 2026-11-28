import { AlertTriangle, Info, ShieldAlert } from "lucide-react";
import type { PendingConfirmation } from "../types";

interface ApprovalModalProps {
  pending: PendingConfirmation | null;
  busy: boolean;
  onDecide: (decision: "confirm" | "cancel") => void;
}

export function ApprovalModal({ pending, busy, onDecide }: ApprovalModalProps) {
  if (!pending) return null;

  const spec = pending.spec || {};
  const severity = spec.severity || (pending.tool_name === "delete_file" ? "danger" : "warning");
  const title = spec.title || `确认执行 ${pending.tool_name}？`;
  const summary =
    spec.summary || "Agent 已暂停执行。只有你明确确认后，该操作才会继续执行。";
  const confirmLabel = spec.confirm_label || "确认执行";
  const cancelLabel = spec.cancel_label || "取消操作";

  const details =
    spec.details && spec.details.length > 0
      ? spec.details
      : Object.entries(pending.arguments || {})
          .filter(([k]) => !k.startsWith("_"))
          .slice(0, 8)
          .map(([label, value]) => ({
            label,
            value: typeof value === "string" ? value : JSON.stringify(value)
          }));

  const eyebrowText =
    severity === "danger"
      ? "HIGH-RISK OPERATION"
      : severity === "info"
        ? "CONFIRMATION REQUIRED"
        : "APPROVAL REQUIRED";

  const confirmBtnClass =
    severity === "danger"
      ? "danger-button"
      : severity === "info"
        ? "primary-button"
        : "warning-button";

  return (
    <div className="modal-backdrop" role="presentation">
      <section
        className={`confirm-modal severity-${severity}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
      >
        <div className={`warning-icon severity-${severity}`}>
          {severity === "danger" ? (
            <ShieldAlert size={20} />
          ) : severity === "info" ? (
            <Info size={20} />
          ) : (
            <AlertTriangle size={20} />
          )}
        </div>
        <span className={`eyebrow ${severity}`}>{eyebrowText}</span>
        <h2 id="confirm-title">{title}</h2>
        <p>{summary}</p>
        <dl>
          {details.map((item, idx) => (
            <div key={`${item.label}-${idx}`}>
              <dt>{item.label}</dt>
              <dd>{item.value}</dd>
            </div>
          ))}
          <div>
            <dt>工具</dt>
            <dd>
              <code>{pending.tool_name}</code>
            </dd>
          </div>
          <div>
            <dt>操作编号</dt>
            <dd>
              <code>{pending.operation_id}</code>
            </dd>
          </div>
          {pending.expires_at && (
            <div>
              <dt>确认有效期</dt>
              <dd>{new Date(pending.expires_at).toLocaleString("zh-CN")}</dd>
            </div>
          )}
        </dl>
        <div className="modal-actions">
          <button
            type="button"
            className="cancel-button"
            onClick={() => onDecide("cancel")}
            disabled={busy}
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            className={confirmBtnClass}
            onClick={() => onDecide("confirm")}
            disabled={busy}
          >
            {busy ? "正在执行…" : confirmLabel}
          </button>
        </div>
      </section>
    </div>
  );
}
