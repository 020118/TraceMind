import { useEffect, useState } from "react";
import { Brain, ChevronDown, ChevronRight, Wrench } from "lucide-react";
import type { ToolTraceItem } from "../types";

export function statusLabel(status: string): string {
  const labels: Record<string, string> = {
    succeeded: "完成",
    failed: "失败",
    timeout: "超时",
    awaiting_confirmation: "等待确认",
    cancelled: "已取消",
    running: "执行中"
  };
  return labels[status.toLowerCase()] || status;
}

export function toolLabel(name: string): string {
  const labels: Record<string, string> = {
    tool_search: "发现可用工具",
    read_file: "读取文件",
    list_dir: "浏览目录",
    write_file: "写入文件",
    edit_file: "修改文件",
    delete_file: "删除文件",
    check_python_syntax: "Python 语法检查",
    run_tests: "运行测试",
    git_status: "查看 Git 状态",
    git_diff: "查看代码 Diff",
    git_create_branch: "创建功能分支",
    code_review_diff: "独立 AI Code Review",
    git_commit_and_push: "提交并推送代码",
    web_search: "搜索网络",
    web_fetch: "读取网页",
    knowledge_search: "检索知识库",
    knowledge_list_documents: "查看知识库"
  };
  return labels[name] || name;
}

function shortResult(result: string): string {
  const clean = result.trim();
  if (clean.length <= 420) return clean;
  return `${clean.slice(0, 420)}…`;
}

interface ThinkingBlockProps {
  reasoning?: string;
  trace: ToolTraceItem[];
  isThinking?: boolean;
  streamStatus?: string;
}

export function ThinkingBlock({
  reasoning,
  trace,
  isThinking,
  streamStatus
}: ThinkingBlockProps) {
  const [open, setOpen] = useState(Boolean(isThinking));

  useEffect(() => {
    if (isThinking) setOpen(true);
  }, [isThinking]);

  const hasReasoning = Boolean(reasoning && reasoning.trim());
  if (!hasReasoning && trace.length === 0 && !isThinking) return null;

  const summary = isThinking
    ? streamStatus || "正在思考与规划…"
    : hasReasoning && trace.length > 0
      ? `思考过程 · ${trace.length} 个执行步骤`
      : hasReasoning
        ? "查看思考过程"
        : `执行过程 · ${trace.length} 个步骤`;

  return (
    <div className={`thinking-block ${isThinking ? "live" : ""}`}>
      <button
        type="button"
        className="thinking-toggle"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
      >
        <span className="thinking-toggle-left">
          <Brain size={14} />
          <span>{summary}</span>
        </span>
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
      </button>
      {open && (
        <div className="thinking-content">
          {hasReasoning && (
            <div className="thinking-reasoning">{reasoning}</div>
          )}
          {trace.length > 0 && (
            <div className="process-list">
              {trace.map((item, index) => (
                <details
                  className={`process-step ${item.status}`}
                  key={`${item.call_id}-${index}`}
                >
                  <summary>
                    <span className="step-index">
                      {String(index + 1).padStart(2, "0")}
                    </span>
                    <strong className="step-name">
                      <Wrench size={11} />
                      {toolLabel(item.name)}
                    </strong>
                    <small>
                      {statusLabel(item.status)} · {item.duration_ms} ms
                    </small>
                  </summary>
                  <div className="step-detail">
                    <label>输入参数</label>
                    <pre>{JSON.stringify(item.arguments, null, 2)}</pre>
                    <label>执行结果</label>
                    <p>{shortResult(item.result) || "无返回内容"}</p>
                  </div>
                </details>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
