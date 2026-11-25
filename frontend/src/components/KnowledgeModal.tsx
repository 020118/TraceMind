import { FileText, Trash2, Upload, X } from "lucide-react";
import type { KnowledgeDocument } from "../types";

interface KnowledgeModalProps {
  open: boolean;
  documents: KnowledgeDocument[];
  busy: boolean;
  onClose: () => void;
  onUpload: (file: File | undefined) => void;
  onDelete: (doc: KnowledgeDocument) => void;
}

export function KnowledgeModal({
  open,
  documents,
  busy,
  onClose,
  onUpload,
  onDelete
}: KnowledgeModalProps) {
  if (!open) return null;

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <section
        className="dialog-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="kb-modal-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <div>
            <span className="eyebrow">KNOWLEDGE BASE</span>
            <h2 id="kb-modal-title">文档知识库管理（{documents.length}）</h2>
          </div>
          <button type="button" className="icon-close" onClick={onClose} aria-label="关闭">
            <X size={18} />
          </button>
        </div>

        <div className="kb-toolbar">
          <p>上传的文本、代码或 PDF 将自动切分并建立 RAG 向量索引，供所有会话检索引用。</p>
          <label className={`primary-btn upload-label ${busy ? "disabled" : ""}`}>
            <Upload size={14} />
            <span>{busy ? "正在索引…" : "上传文档"}</span>
            <input
              type="file"
              accept=".txt,.md,.markdown,.json,.yaml,.yml,.toml,.py,.js,.ts,.tsx,.jsx,.csv,.pdf"
              disabled={busy}
              onChange={(event) => {
                onUpload(event.target.files?.[0]);
                event.currentTarget.value = "";
              }}
            />
          </label>
        </div>

        <div className="kb-list">
          {documents.length === 0 ? (
            <div className="kb-empty">暂无已索引文档。点击右上角“上传文档”开始构建个人知识库。</div>
          ) : (
            documents.map((doc) => (
              <div className="kb-row" key={doc.id}>
                <div className="kb-row-icon">
                  <FileText size={16} />
                </div>
                <div className="kb-row-info">
                  <strong title={doc.source_path}>{doc.title}</strong>
                  <small>
                    {doc.chunk_count} 个片段 · {doc.media_type} · 向量状态：{doc.vector_status || "ready"}
                  </small>
                </div>
                <button
                  type="button"
                  className="kb-delete-btn"
                  onClick={() => onDelete(doc)}
                  aria-label={`移除 ${doc.title}`}
                  title="移除文档"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))
          )}
        </div>
      </section>
    </div>
  );
}
