import { FormEvent, useState } from "react";
import { ChatResponse, LeadInput, sendQuestion, submitContact, submitLead } from "./api";

type Message = { role: "assistant" | "user"; text: string; result?: ChatResponse };

const SESSION_EMAIL_KEY = "jhs-parent-follow-up-email";

function getSessionEmail() {
  try {
    return sessionStorage.getItem(SESSION_EMAIL_KEY) ?? "";
  } catch {
    return "";
  }
}

const DETAIL_LABELS: Record<string, string> = {
  pickup_area: "Pickup area / locality",
  pickup_location: "Pickup area / locality",
  location: "Locality or address",
  grade: "Grade / class",
  program: "Program",
  academic_year: "Academic year",
  admission_year: "Admission year",
  date_of_birth: "Child’s date of birth",
  child_age: "Child’s age",
};

function detailLabel(detail: string) {
  return DETAIL_LABELS[detail] ?? detail.replaceAll("_", " ").replace(/^./, (value) => value.toUpperCase());
}

export function App() {
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", text: "Hello! How can I help you with Jain Heritage School today?" },
  ]);
  const [question, setQuestion] = useState("");
  const [email, setEmail] = useState(getSessionEmail);
  const [rememberedEmail, setRememberedEmail] = useState(getSessionEmail);
  const [contactedQuestions, setContactedQuestions] = useState<Set<string>>(new Set());
  const [contactPending, setContactPending] = useState<string>();
  const [contactError, setContactError] = useState<string>();
  const [detailValues, setDetailValues] = useState<Record<number, Record<string, string>>>({});
  const [conversationId, setConversationId] = useState<string>();
  const [pending, setPending] = useState(false);
  const [showLeadForm, setShowLeadForm] = useState(false);
  const [lead, setLead] = useState<LeadInput>({
    student_name: "", grade_applied_for: "", guardian_name: "",
    mobile_number: "", email: "", preferred_contact: "", consent_to_contact: false,
  });
  async function ask(event: FormEvent) {
    event.preventDefault();
    const value = question.trim();
    if (!value || pending) return;
    setQuestion("");
    await askQuestion(value);
  }

  async function askQuestion(
    value: string,
    forceEscalation = false,
    displayText = value,
    context: Record<string, string> = {},
  ) {
    if (!value.trim() || pending) return;
    setMessages((current) => [...current, { role: "user", text: displayText }]);
    setPending(true);
    try {
      const result = await sendQuestion(value, conversationId, forceEscalation, context);
      setConversationId(result.conversation_id);
      setMessages((current) => [...current, { role: "assistant", text: result.answer, result }]);
      const rememberedEmail = getSessionEmail();
      if (result.outcome === "escalated" && result.unanswered_id && rememberedEmail) {
        try {
          await submitContact(result.unanswered_id, rememberedEmail);
          setContactedQuestions((current) => new Set(current).add(result.unanswered_id!));
        } catch {
          // The question is already in the school queue. Keep the inline form available
          // so the parent can retry attaching their email without losing the enquiry.
        }
      }
    } catch (error) {
      setMessages((current) => [...current, { role: "assistant", text: (error as Error).message }]);
    } finally {
      setPending(false);
    }
  }

  async function submitClarificationDetails(
    event: FormEvent,
    messageIndex: number,
    result: ChatResponse,
  ) {
    event.preventDefault();
    const context = detailValues[messageIndex] ?? {};
    if (result.required_details.some((detail) => !context[detail]?.trim())) return;
    const displayText = result.required_details
      .map((detail) => `${detailLabel(detail)}: ${context[detail].trim()}`)
      .join(" · ");
    await askQuestion(
      result.clarification_question ?? "School enquiry",
      false,
      displayText,
      context,
    );
  }

  async function shareEmail(event: FormEvent, unansweredId: string) {
    event.preventDefault();
    const normalizedEmail = email.trim().toLowerCase();
    if (!normalizedEmail || contactPending) return;
    setContactPending(unansweredId);
    setContactError(undefined);
    try {
      const unansweredIds = new Set(
        messages
          .map((message) => message.result?.unanswered_id)
          .filter((id): id is string => Boolean(id)),
      );
      unansweredIds.add(unansweredId);
      const idsToAttach = [...unansweredIds].filter((id) => !contactedQuestions.has(id));
      await Promise.all(idsToAttach.map((id) => submitContact(id, normalizedEmail)));
      sessionStorage.setItem(SESSION_EMAIL_KEY, normalizedEmail);
      setEmail(normalizedEmail);
      setRememberedEmail(normalizedEmail);
      setContactedQuestions((current) => new Set([...current, ...idsToAttach]));
    } catch (error) {
      setContactError((error as Error).message);
    } finally {
      setContactPending(undefined);
    }
  }

  async function sendLead(event: FormEvent) {
    event.preventDefault();
    setPending(true);
    try {
      await submitLead(lead);
      setShowLeadForm(false);
      setMessages((current) => [...current, {
        role: "assistant",
        text: "Thank you. Your admission enquiry has been submitted to the JHS admissions team.",
      }]);
    } catch (error) {
      setMessages((current) => [...current, { role: "assistant", text: (error as Error).message }]);
    } finally {
      setPending(false);
    }
  }

  const suggestions = ["What grades are offered?", "What facilities are available?", "Where is the school located?"];

  return (
    <main className="page-shell">
      <section className="intro">
        <span className="eyebrow">Jain Heritage School</span>
        <h1>Discover a joyful way to learn.</h1>
        <p>Ask about academics, programmes, facilities, campus visits, transport, or admissions.</p>
      </section>

      <section className="chat-card" aria-label="JHS school information chat">
        <header><div className="crest">JGI</div><div><strong>JHS School Assistant</strong><span>Answers from approved school information</span></div></header>
        <div className="messages" aria-live="polite">
          {messages.map((message, index) => (
            <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
              <p>{message.text}</p>
              {message.result && message.result.citations.length > 0 && (
                <details className="message-sources">
                  <summary>
                    View {message.result.citations.length} {message.result.citations.length === 1 ? "source" : "sources"}
                  </summary>
                  <div className="source-list">
                    {message.result.citations.map((citation, citationIndex) => (
                      <div className="source-item" key={`${citation.knowledge_id}-${citationIndex}`}>
                        <strong>{citation.title}</strong>
                        <small>{citation.source_label}</small>
                      </div>
                    ))}
                  </div>
                </details>
              )}
              {message.result?.outcome === "clarification" && message.result.clarification_kind !== "details" && (
                <div className="clarification-options">
                  {message.result.suggested_questions.map((suggestedQuestion) => (
                    <button type="button" disabled={pending} onClick={() => void askQuestion(suggestedQuestion)} key={suggestedQuestion}>{suggestedQuestion}</button>
                  ))}
                  <button type="button" className="none-option" disabled={pending} onClick={() => void askQuestion(message.result?.clarification_question ?? "", true, "None of these")}>None of these</button>
                </div>
              )}
              {message.result?.outcome === "clarification" && message.result.clarification_kind === "details" && (
                <form className="detail-clarification" onSubmit={(event) => void submitClarificationDetails(event, index, message.result!)}>
                  {message.result.required_details.map((detail) => (
                    <label key={detail}>
                      {detailLabel(detail)}
                      <input
                        required
                        type={detail === "date_of_birth" ? "date" : "text"}
                        value={detailValues[index]?.[detail] ?? ""}
                        onChange={(event) => setDetailValues((current) => ({
                          ...current,
                          [index]: { ...current[index], [detail]: event.target.value },
                        }))}
                        placeholder={detail.includes("location") || detail.includes("area") ? "Example: JP Nagar, Bengaluru" : undefined}
                      />
                    </label>
                  ))}
                  <button disabled={pending}>Check with these details</button>
                  <small>For transport, enter only the area or locality—not a house number. We use these details only to answer this enquiry.</small>
                </form>
              )}
              {message.result?.outcome === "escalated" && message.result.unanswered_id && (
                contactedQuestions.has(message.result.unanswered_id) ? (
                  <div className="follow-up-saved">
                    <span aria-hidden="true">✓</span>
                    <div><strong>Email saved</strong><small>The school can reply to {rememberedEmail}.</small></div>
                  </div>
                ) : (
                  <form className="inline-contact-form" onSubmit={(event) => void shareEmail(event, message.result!.unanswered_id!)}>
                    <label htmlFor={`email-${message.result.unanswered_id}`}>Would you like the school to reply to you?</label>
                    <p className="contact-help">Your question is already in the school’s review queue. Email is optional.</p>
                    <div>
                      <input
                        id={`email-${message.result.unanswered_id}`}
                        type="email"
                        autoComplete="email"
                        placeholder="you@example.com"
                        value={email}
                        onChange={(event) => setEmail(event.target.value)}
                        required
                      />
                      <button disabled={contactPending === message.result.unanswered_id}>
                        {contactPending === message.result.unanswered_id ? "Saving…" : "Notify me"}
                      </button>
                    </div>
                    <small>We’ll remember this email only for this browser session.</small>
                    {contactError && <small className="contact-error">{contactError}</small>}
                  </form>
                )
              )}
            </article>
          ))}
          {pending && <article className="message assistant"><p>Checking approved school information…</p></article>}
        </div>

        <div className="quick-actions">
          {suggestions.map((suggestion) => (
            <button type="button" key={suggestion} onClick={() => setQuestion(suggestion)}>{suggestion}</button>
          ))}
          <button type="button" className="primary-chip" onClick={() => setShowLeadForm(true)}>Admission enquiry</button>
        </div>

        {showLeadForm && (
          <form className="lead-form" onSubmit={sendLead}>
            <div className="form-heading"><strong>Admission enquiry</strong><button type="button" onClick={() => setShowLeadForm(false)} aria-label="Close">×</button></div>
            <div className="form-grid">
              <label>Student’s full name<input required value={lead.student_name} onChange={(e) => setLead({...lead, student_name: e.target.value})} /></label>
              <label>Grade / class<input required value={lead.grade_applied_for} onChange={(e) => setLead({...lead, grade_applied_for: e.target.value})} /></label>
              <label>Parent / guardian<input required value={lead.guardian_name} onChange={(e) => setLead({...lead, guardian_name: e.target.value})} /></label>
              <label>Mobile number<input required type="tel" value={lead.mobile_number} onChange={(e) => setLead({...lead, mobile_number: e.target.value})} /></label>
              <label>Email address<input required type="email" value={lead.email} onChange={(e) => setLead({...lead, email: e.target.value})} /></label>
              <label>Preferred contact time / location<input value={lead.preferred_contact} onChange={(e) => setLead({...lead, preferred_contact: e.target.value})} /></label>
            </div>
            <label className="consent"><input required type="checkbox" checked={lead.consent_to_contact} onChange={(e) => setLead({...lead, consent_to_contact: e.target.checked})} /> I consent to being contacted about this enquiry.</label>
            <button disabled={pending}>Submit enquiry</button>
          </form>
        )}

        <form className="composer" onSubmit={ask}>
          <input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask an admissions question…" aria-label="Your question" />
          <button disabled={pending}>Ask</button>
        </form>
      </section>
    </main>
  );
}
