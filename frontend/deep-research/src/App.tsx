
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Search,
  FileText,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Download,
  X,
} from "lucide-react";
import "./App.css";

type TaskStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed";

interface TodoItem {
  id: number;
  title: string;
  intent: string;
  query: string;
  status: TaskStatus;
}

interface ResearchEvent {
  type: string;
  message?: string;
  task_id?: number;
  progress?: number;
  data?: {
    tasks?: TodoItem[];
    task?: TodoItem;
    summary?: string;
    report?: string;
    research_id?: string;
    error?: string | null;
  };
}

const API_URL = "http://localhost:8000";

export default function App() {
  const [topic, setTopic] = useState("");
  const [maxTasks, setMaxTasks] = useState(3);
  const [tasks, setTasks] = useState<TodoItem[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [report, setReport] = useState("");
  const [progress, setProgress] = useState(0);
  const [loading, setLoading] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [error, setError] = useState("");
  const [researchId, setResearchId] = useState("");

  const abortController = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => abortController.current?.abort();
  }, []);

  function addLog(message: string) {
    setLogs((previous) => [...previous, message]);
  }

  function handleEvent(event: ResearchEvent) {
    if (event.message) addLog(event.message);

    if (event.progress !== undefined) {
      setProgress(event.progress);
    }

    switch (event.type) {
      case "tasks":
        setTasks(event.data?.tasks ?? []);
        break;

      case "task":
        if (event.data?.task) {
          const updated = event.data.task;

          setTasks((previous) =>
            previous.map((task) =>
              task.id === updated.id ? updated : task
            )
          );
        }
        break;

      case "report":
        setReport(event.data?.report ?? "");
        setResearchId(event.data?.research_id ?? "");
        setProgress(100);
        setLoading(false);
        break;

      case "error":
        setError(event.message ?? "Research failed");
        setLoading(false);
        break;
    }
  }

  async function startResearch() {
    if (topic.trim().length < 3 || loading) return;

    abortController.current?.abort();

    const controller = new AbortController();
    abortController.current = controller;

    setTasks([]);
    setLogs([]);
    setReport("");
    setProgress(0);
    setError("");
    setResearchId("");
    setLoading(true);
    setModalOpen(true);

    try {
      const response = await fetch(
        `${API_URL}/research/stream`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            topic: topic.trim(),
            max_tasks: maxTasks,
            max_results_per_task: 5,
          }),
          signal: controller.signal,
        }
      );

      if (!response.ok || !response.body) {
        throw new Error(
          `Backend returned HTTP ${response.status}`
        );
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE events are separated by blank lines.
        const chunks = buffer.split(/\r?\n\r?\n/);
        buffer = chunks.pop() ?? "";

        for (const chunk of chunks) {
          const data = chunk
            .split(/\r?\n/)
            .filter((line) => line.startsWith("data:"))
            .map((line) => line.slice(5).trimStart())
            .join("\n");

          if (!data) continue;

          const event = JSON.parse(data) as ResearchEvent;
          handleEvent(event);
        }
      }

      if (!controller.signal.aborted) {
        setLoading(false);
      }
    } catch (err) {
      if (controller.signal.aborted) return;

      setError(
        err instanceof Error
          ? err.message
          : "Unexpected research error"
      );
      setLoading(false);
    }
  }

  function closeModal() {
    abortController.current?.abort();
    setLoading(false);
    setModalOpen(false);
  }

  function downloadReport() {
    if (!report) return;

    const blob = new Blob([report], {
      type: "text/markdown;charset=utf-8",
    });

    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");

    anchor.href = url;
    anchor.download = `research-${researchId || "report"}.md`;
    anchor.click();

    URL.revokeObjectURL(url);
  }

  return (
    <div className="app">
      <header className="site-header">
        <div className="brand">
          <Search size={23} />
          <span>Deep Research Agent</span>
        </div>
        <span className="header-caption">
          Powered by Ahmad Marei
        </span>
      </header>

      <main className="hero">
        <div className="eyebrow">
          AUTOMATED DEEP RESEARCH
        </div>

        <h1>Research complex topics with AI</h1>

        <p className="subtitle">
          Break down questions, collect web sources,
          and generate structured research reports.
        </p>

        <div className="search-panel">
          <textarea
            value={topic}
            onChange={(event) =>
              setTopic(event.target.value)
            }
            placeholder="What would you like to research?"
            rows={4}
          />

          <div className="search-actions">
            <label>
              Research tasks
              <select
                value={maxTasks}
                onChange={(event) =>
                  setMaxTasks(Number(event.target.value))
                }
              >
                <option value={3}>3 tasks</option>
                <option value={4}>4 tasks</option>
                <option value={5}>5 tasks</option>
              </select>
            </label>

            <button
              className="primary-button"
              disabled={topic.trim().length < 3}
              onClick={startResearch}
            >
              <Search size={18} />
              Start Research
            </button>
          </div>
        </div>
      </main>

      {modalOpen && (
        <div className="modal-overlay">
          <div className="research-modal">
            <div className="modal-header">
              <div>
                <h2>{topic}</h2>
                <span className="muted">
                  {loading
                    ? "Research in progress"
                    : error
                    ? "Research encountered an error"
                    : "Research session"}
                </span>
              </div>

              <button
                className="icon-button"
                onClick={closeModal}
                aria-label="Close research"
              >
                <X size={22} />
              </button>
            </div>

            <div className="progress-section">
              <div className="progress-label">
                <span>Research progress</span>
                <strong>{progress}%</strong>
              </div>

              <div className="progress-track">
                <div
                  className="progress-fill"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>

            <div className="research-content">
              <aside className="sidebar">
                <h3>Research tasks</h3>

                {tasks.length === 0 && (
                  <p className="muted">
                    Waiting for the research plan...
                  </p>
                )}

                {tasks.map((task) => (
                  <div className="task-card" key={task.id}>
                    <div className="task-title">
                      {task.status === "completed" ? (
                        <CheckCircle2
                          size={18}
                          className="success"
                        />
                      ) : task.status === "failed" ? (
                        <AlertCircle
                          size={18}
                          className="error"
                        />
                      ) : task.status === "running" ? (
                        <Loader2
                          size={18}
                          className="spinner"
                        />
                      ) : (
                        <span className="pending-dot" />
                      )}

                      <strong>{task.title}</strong>
                    </div>

                    <p>{task.intent}</p>
                    <small>{task.status}</small>
                  </div>
                ))}

                <h3 className="activity-heading">
                  Activity
                </h3>

                <div className="activity-log">
                  {logs.map((log, index) => (
                    <p key={index}>{log}</p>
                  ))}
                </div>
              </aside>

              <section className="report-section">
                <div className="report-header">
                  <h3>
                    <FileText size={20} />
                    Research report
                  </h3>

                  {report && (
                    <button
                      className="secondary-button"
                      onClick={downloadReport}
                    >
                      <Download size={16} />
                      Download Markdown
                    </button>
                  )}
                </div>

                {error && (
                  <div className="error-banner">
                    <AlertCircle size={18} />
                    {error}
                  </div>
                )}

                {report ? (
                  <article className="markdown-body">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                    >
                      {report}
                    </ReactMarkdown>
                  </article>
                ) : (
                  <div className="report-placeholder">
                    {loading ? (
                      <Loader2
                        size={34}
                        className="spinner"
                      />
                    ) : (
                      <FileText size={34} />
                    )}

                    <p>
                      {loading
                        ? "Collecting evidence and preparing your report..."
                        : "No report available."}
                    </p>
                  </div>
                )}
              </section>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
