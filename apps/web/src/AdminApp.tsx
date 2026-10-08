import { FormEvent, useEffect, useState } from "react";
import {
  AdminQuestion,
  AdminSuggestion,
  Citation,
  approveAdminAnswer,
  generateAdminSuggestions,
  getAdminQuestion,
  listAdminQuestions,
  retryAdminEmail,
  saveAdminDraft,
} from "./api";

type QueueStatus = AdminQuestion["status"];

function splitQuestionContext(rawQuestion: string) {
  const [question, detailsBlock = ""] = rawQuestion.split(/\n\s*ADDITIONAL DETAILS\s*\n/i, 2);
  const details = detailsBlock
    .split("\n")
    .map((line) => line.replace(/^\s*-\s*/, "").trim())
    .filter(Boolean)
    .map((line) => {
      const separator = line.indexOf(":");
      return separator < 0
        ? { label: "Additional detail", value: line }
        : { label: line.slice(0, separator).trim(), value: line.slice(separator + 1).trim() };
    });
  return { question: question.trim(), details };
}

function reusableKnowledgeTitle(rawQuestion: string) {
  const parsed = splitQuestionContext(rawQuestion);
  const reusableLabels = new Set([
    "pickup area",
    "pickup location",
    "grade",
    "program",
    "academic year",
    "admission year",
  ]);
  const reusableDetails = parsed.details.filter((detail) =>
    reusableLabels.has(detail.label.toLowerCase())
  );
  const suffix = reusableDetails
    .map((detail) => `${detail.label}: ${detail.value}`)
    .join(" · ");
  return `${parsed.question}${suffix ? ` — ${suffix}` : ""}`.slice(0, 200);
}

export function AdminApp() {
  const [adminKey, setAdminKey] = useState(() => sessionStorage.getItem("jhs-admin-key") ?? "");
  const [keyInput, setKeyInput] = useState(adminKey);
  const [status, setStatus] = useState<QueueStatus>("open");
  const [questions, setQuestions] = useState<AdminQuestion[]>([]);
  const [selected, setSelected] = useState<AdminQuestion>();
  const [suggestions, setSuggestions] = useState<AdminSuggestion[]>([]);
  const [answer, setAnswer] = useState("");
  const [selectedSources, setSelectedSources] = useState<Citation[]>([]);
  const [publishToKnowledge, setPublishToKnowledge] = useState(true);
  const [knowledgeTitle, setKnowledgeTitle] = useState("");
  const [validFrom, setValidFrom] = useState("");
  const [expiresAt, setExpiresAt] = useState("");
  const [supersedeIds, setSupersedeIds] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const hasUnfilledPlaceholders = /\[\[[^\]]+\]\]/.test(answer);
  const availableSources = Array.from(
    new Map(
      suggestions.flatMap((suggestion) => suggestion.citations).map((source) => [source.knowledge_id, source]),
    ).values(),
  );

  async function loadQueue(key = adminKey, queueStatus = status) {
    if (!key) return;
    setLoading(true);
    setError("");
    try {
      const items = await listAdminQuestions(key, queueStatus);
      setQuestions(items);
      if (selected && !items.some((item) => item.id === selected.id)) {
        setSelected(undefined);
        setAnswer("");
        setSuggestions([]);
        resetPublication();
      }
    } catch (requestError) {
      setError((requestError as Error).message);
      if ((requestError as Error).name === "UnauthorizedError") logout();
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (adminKey) void loadQueue(adminKey, status);
    // The queue is intentionally refreshed when credentials or the filter changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [adminKey, status]);

  async function login(event: FormEvent) {
    event.preventDefault();
    const value = keyInput.trim();
    if (!value) return;
    sessionStorage.setItem("jhs-admin-key", value);
    setAdminKey(value);
  }

  function logout() {
    sessionStorage.removeItem("jhs-admin-key");
    setAdminKey("");
    setKeyInput("");
    setQuestions([]);
    setSelected(undefined);
    setSuggestions([]);
    setAnswer("");
    resetPublication();
  }

  function resetPublication() {
    setSelectedSources([]);
    setPublishToKnowledge(true);
    setKnowledgeTitle("");
    setValidFrom("");
    setExpiresAt("");
    setSupersedeIds([]);
  }

  async function chooseQuestion(item: AdminQuestion) {
    setLoading(true);
    setError("");
    setNotice("");
    try {
      const detail = await getAdminQuestion(adminKey, item.id);
      setSelected(detail);
      setAnswer(detail.latest_answer ?? "");
      setSuggestions([]);
      resetPublication();
      setKnowledgeTitle(reusableKnowledgeTitle(detail.question));
    } catch (requestError) {
      setError((requestError as Error).message);
    } finally {
      setLoading(false);
    }
  }

  async function requestSuggestions() {
    if (!selected) return;
    setLoading(true);
    setError("");
    setNotice("");
    try {
      const result = await generateAdminSuggestions(adminKey, selected.id);
      setSuggestions(result.suggestions);
      setNotice("Draft suggestions are ready for staff review.");
    } catch (requestError) {
      setError((requestError as Error).message);
    } finally {
      setLoading(false);
    }
  }

  async function saveDraft() {
    if (!selected || answer.trim().length < 2) return;
    setLoading(true);
    setError("");
    try {
      await saveAdminDraft(adminKey, selected.id, answer.trim());
      const detail = await getAdminQuestion(adminKey, selected.id);
      setSelected(detail);
      setNotice("Draft saved. It has not been sent to the parent.");
      await loadQueue();
    } catch (requestError) {
      setError((requestError as Error).message);
    } finally {
      setLoading(false);
    }
  }

  async function approve() {
    if (!selected || answer.trim().length < 2) return;
    if (hasUnfilledPlaceholders) {
      setError("Replace every [[PLACEHOLDER]] with verified school information before approval.");
      return;
    }
    const confirmed = window.confirm(
      "Approve this verified answer? It will be emailed to the parent when an email is available.",
    );
    if (!confirmed) return;
    setLoading(true);
    setError("");
    try {
      if (publishToKnowledge && knowledgeTitle.trim().length < 2) {
        setError("Enter a reusable knowledge title before publishing.");
        return;
      }
      const result = await approveAdminAnswer(adminKey, selected.id, answer.trim(), {
        publish_to_knowledge: publishToKnowledge,
        knowledge_title: publishToKnowledge ? knowledgeTitle.trim() : undefined,
        valid_from: publishToKnowledge && validFrom ? `${validFrom}T00:00:00Z` : undefined,
        expires_at: publishToKnowledge && expiresAt ? `${expiresAt}T23:59:59Z` : undefined,
        supersede_knowledge_ids: publishToKnowledge ? supersedeIds : [],
      });
      const knowledgeMessage = result.knowledge_entry_id
        ? "Answer approved and published to the knowledge base."
        : "Answer approved and stored.";
      const deliveryMessage = result.delivery_status === "sent"
        ? " The parent email was sent."
        : result.delivery_status === "failed"
          ? " The email could not be sent; staff can retry from the answered question."
          : " No parent email was available, so no message was sent.";
      setNotice(`${knowledgeMessage}${deliveryMessage}`);
      setSelected(undefined);
      setSuggestions([]);
      setAnswer("");
      resetPublication();
      await loadQueue();
    } catch (requestError) {
      setError((requestError as Error).message);
    } finally {
      setLoading(false);
    }
  }

  async function retryEmail() {
    if (!selected) return;
    setLoading(true);
    setError("");
    setNotice("");
    try {
      const detail = await retryAdminEmail(adminKey, selected.id);
      setSelected(detail);
      if (detail.delivery_status === "sent") {
        setNotice("The approved answer was emailed to the parent.");
      } else {
        setError(detail.delivery_error ?? "The email could not be sent.");
      }
      await loadQueue();
    } catch (requestError) {
      setError((requestError as Error).message);
    } finally {
      setLoading(false);
    }
  }

  if (!adminKey) {
    return (
      <main className="admin-login-shell">
        <form className="admin-login" onSubmit={login}>
          <span className="eyebrow">JHS Electronic City</span>
          <h1>School response desk</h1>
          <p>Enter the private admin key to review parent questions.</p>
          <label>Admin key<input type="password" value={keyInput} onChange={(event) => setKeyInput(event.target.value)} autoComplete="current-password" /></label>
          <button type="submit">Open response desk</button>
          {error && <p className="admin-error">{error}</p>}
          <a href="/">Return to admissions assistant</a>
        </form>
      </main>
    );
  }

  return (
    <main className="admin-shell">
      <header className="admin-topbar">
        <div><span className="eyebrow">Jain Heritage School</span><strong>Response desk</strong></div>
        <nav><a href="/">Public assistant</a><button type="button" className="secondary-button" onClick={logout}>Sign out</button></nav>
      </header>

      <section className="admin-queue">
        <div className="queue-heading">
          <div><h2>Parent questions</h2><p>{questions.length} questions in this view</p></div>
          <button type="button" className="secondary-button" onClick={() => void loadQueue()} disabled={loading}>Refresh</button>
        </div>
        <div className="status-tabs" role="tablist">
          {(["open", "answered", "closed"] as QueueStatus[]).map((value) => (
            <button type="button" className={status === value ? "active" : ""} onClick={() => setStatus(value)} key={value}>{value}</button>
          ))}
        </div>
        <div className="question-list">
          {!loading && questions.length === 0 && <p className="empty-state">No {status} questions.</p>}
          {questions.map((item) => (
            <button type="button" className={`question-row ${selected?.id === item.id ? "selected" : ""}`} onClick={() => void chooseQuestion(item)} key={item.id}>
              <span>{splitQuestionContext(item.question).question}</span>
              {splitQuestionContext(item.question).details.length > 0 && <small className="context-available">Parent details included</small>}
              <small>{new Date(item.created_at).toLocaleString()} · {item.contact_email ? "Email provided" : "No email"}</small>
            </button>
          ))}
        </div>
      </section>

      <section className="admin-workspace">
        {!selected ? (
          <div className="workspace-placeholder"><h2>Select a question</h2><p>Review its details, prepare a draft, and approve only verified information.</p></div>
        ) : (
          <>
            <div className="question-detail">
              <span className={`status-pill ${selected.status}`}>{selected.status}</span>
              <h2>{splitQuestionContext(selected.question).question}</h2>
              {splitQuestionContext(selected.question).details.length > 0 && (
                <section className="parent-context">
                  <strong>Parent-provided details</strong>
                  <dl>
                    {splitQuestionContext(selected.question).details.map((detail) => (
                      <div key={`${detail.label}-${detail.value}`}><dt>{detail.label}</dt><dd>{detail.value}</dd></div>
                    ))}
                  </dl>
                </section>
              )}
              <dl><div><dt>Parent email</dt><dd>{selected.contact_email ?? "Not provided"}</dd></div><div><dt>Received</dt><dd>{new Date(selected.created_at).toLocaleString()}</dd></div></dl>
            </div>

            {selected.status === "open" && (
              <>
                <div className="suggestion-heading"><div><h3>AI-assisted drafts</h3><p>Suggestions are never sent automatically.</p></div><button type="button" onClick={() => void requestSuggestions()} disabled={loading}>Generate suggestions</button></div>
                <div className="suggestion-list">
                  {suggestions.map((suggestion, index) => (
                    <article className="suggestion-card" key={`${suggestion.kind}-${index}`}>
                      <div><span className={`suggestion-kind ${suggestion.kind}`}>{suggestion.kind}</span><span>Staff verification required</span></div>
                      <p>{suggestion.answer}</p>
                      {suggestion.citations.map((citation) => <small key={citation.knowledge_id}>Source: {citation.source_label}</small>)}
                      <button type="button" className="secondary-button" onClick={() => { setAnswer(suggestion.answer); setSelectedSources(availableSources); setSupersedeIds([]); }}>Use this draft</button>
                    </article>
                  ))}
                </div>

                <label className="answer-editor">Final response<textarea value={answer} onChange={(event) => setAnswer(event.target.value)} placeholder="Select a suggestion or write a verified response…" rows={8} /></label>
                {hasUnfilledPlaceholders && <p className="placeholder-warning">Complete every highlighted [[PLACEHOLDER]] before approval. Saving an incomplete draft is allowed.</p>}
                <div className="review-warning"><strong>Staff responsibility</strong><span>Verify every name, date, fee, availability statement and policy before approval.</span></div>
                <section className="publication-panel">
                  <label className="publish-toggle"><input type="checkbox" checked={publishToKnowledge} onChange={(event) => setPublishToKnowledge(event.target.checked)} /> Make this approved response reusable in the knowledge base</label>
                  {publishToKnowledge && (
                    <div className="publication-fields">
                      <label>Knowledge title<input value={knowledgeTitle} onChange={(event) => setKnowledgeTitle(event.target.value)} placeholder="Example: School timings 2026–27" /></label>
                      <label>Valid from <span className="optional-label">Optional</span><input type="date" value={validFrom} onChange={(event) => setValidFrom(event.target.value)} /></label>
                      <label>Expires on <span className="optional-label">Optional</span><input type="date" value={expiresAt} onChange={(event) => setExpiresAt(event.target.value)} /></label>
                      <p className="validity-help">Leave both dates empty for information that remains valid indefinitely. Use an expiry date for routes, fees, availability, or other time-sensitive answers.</p>
                      {selectedSources.length > 0 && <fieldset><legend>Archive only sources confirmed to be outdated</legend>{selectedSources.map((source) => <label key={source.knowledge_id}><input type="checkbox" checked={supersedeIds.includes(source.knowledge_id)} onChange={(event) => setSupersedeIds((current) => event.target.checked ? [...current, source.knowledge_id] : current.filter((id) => id !== source.knowledge_id))} /> {source.title} — {source.source_label}</label>)}</fieldset>}
                    </div>
                  )}
                </section>
                <div className="admin-actions"><button type="button" className="secondary-button" onClick={() => void saveDraft()} disabled={loading || answer.trim().length < 2}>Save draft</button><button type="button" onClick={() => void approve()} disabled={loading || answer.trim().length < 2 || hasUnfilledPlaceholders}>Approve answer</button></div>
              </>
            )}

            {selected.status !== "open" && selected.latest_answer && (
              <div className="approved-answer">
                <h3>Latest approved response</h3>
                <p>{selected.latest_answer}</p>
                <div className={`delivery-status ${selected.delivery_status ?? "not_applicable"}`}>
                  <strong>Email delivery: {(selected.delivery_status ?? "not_applicable").replace("_", " ")}</strong>
                  {selected.delivered_at && <span>Sent {new Date(selected.delivered_at).toLocaleString()}</span>}
                  {selected.delivery_error && <span>{selected.delivery_error}</span>}
                  {(selected.delivery_status === "failed" || selected.delivery_status === "pending") && selected.contact_email && (
                    <button type="button" onClick={() => void retryEmail()} disabled={loading}>Retry email</button>
                  )}
                </div>
              </div>
            )}
          </>
        )}
        {loading && <div className="admin-loading">Working…</div>}
        {notice && <p className="admin-notice">{notice}</p>}
        {error && <p className="admin-error">{error}</p>}
      </section>
    </main>
  );
}
