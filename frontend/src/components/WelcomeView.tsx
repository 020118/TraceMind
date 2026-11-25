import { BookOpen, Code2, Image as ImageIcon, Search } from "lucide-react";

const SUGGESTIONS = [
  {
    icon: Code2,
    title: "阅读与重构代码",
    desc: "梳理当前工作区架构，定位关键瓶颈并给出修改方案",
    prompt: "帮我阅读当前项目目录结构，总结核心模块职责并指出最值得优化的 3 个点。"
  },
  {
    icon: BookOpen,
    title: "知识库问答与引用",
    desc: "基于已上传的 PDF、笔记或代码文档做精准检索总结",
    prompt: "列出当前知识库中已经索引的文档，并帮我概括它们的核心主题。"
  },
  {
    icon: Search,
    title: "实时网络调研",
    desc: "联网检索最新技术资料并整理成结构化对比表格",
    prompt: "帮我搜索并整理近期开源 Agent 框架的主流设计模式，用 Markdown 表格对比优缺点。"
  },
  {
    icon: ImageIcon,
    title: "图片与附件分析",
    desc: "直接在输入框粘贴截图或上传文件，让我帮你解读细节",
    prompt: "我想发给你一张截图或文档，请告诉我你支持哪些多模态与代码分析能力？"
  }
];

interface WelcomeViewProps {
  avatarSrc: string;
  onSelectPrompt: (prompt: string) => void;
}

export function WelcomeView({ avatarSrc, onSelectPrompt }: WelcomeViewProps) {
  return (
    <div className="welcome-stage">
      <div className="welcome-hero">
        <img className="welcome-avatar" src={avatarSrc} alt="TraceMind" />
        <span className="eyebrow">TRACEMIND · PERSONAL AGENT WORKSPACE</span>
        <h2>今天想推进什么？</h2>
        <p>
          支持多步工具调用、长期记忆、文档知识库检索、实时思考流与图片附件分析。
        </p>
      </div>
      <div className="suggestion-grid">
        {SUGGESTIONS.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.title}
              type="button"
              className="suggestion-card"
              onClick={() => onSelectPrompt(item.prompt)}
            >
              <div className="suggestion-icon">
                <Icon size={16} />
              </div>
              <strong>{item.title}</strong>
              <span>{item.desc}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
