import { FormEvent, useState } from "react";
import { ChatResponse, LeadInput, sendQuestion, submitContact, submitLead } from "./api";

type Message = { role: "assistant" | "user"; text: string; result?: ChatResponse };

export function App() {
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", text: "Hello! How can I help with admissions today?" },
  ]);
  const [question, setQuestion] = useState("");
  const [email, setEmail] = useState("");
  const [conversationId, setConversationId] = useState<string>();
  const [pending, setPending] = useState(false);
  const [showLeadForm, setShowLeadForm] = useState(false);
  const [lead, setLead] = useState<LeadInput>({
    student_name: "", grade_applied_for: "", guardian_name: "",
    mobile_number: "", email: "", preferred_contact: "", consent_to_contact: false,
  });
  const latestEscalation = [...messages].reverse().find((message) => message.result?.unanswered_id)?.result;

  async function ask(event: FormEvent) {
    event.preventDefault();
    const value = question.trim();
    if (!value || pending) return;
    setQuestion("");
    setMessages((current) => [...current, { role: "user", text: value }]);
    setPending(true);
    try {
      const result = await sendQuestion(value, conversationId);
      setConversationId(result.conversation_id);
      setMessages((current) => [...current, { role: "assistant", text: result.answer, result }]);
    } catch (error) {
      setMessages((current) => [...current, { role: "assistant", text: (error as Error).message }]);
    } finally {
      setPending(false);
    }
  }

  async function shareEmail(event: FormEvent) {
    event.preventDefault();
    if (!latestEscalation?.unanswered_id) return;
    await submitContact(latestEscalation.unanswered_id, email);
    setEmail("");
    setMessages((current) => [
      ...current,
      { role: "assistant", text: "Thank you. Your question has been sent to the admissions team." },
    ]);
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
        <p>Ask about grades, timings, applications, facilities, or admission requirements.</p>
      </section>

      <section className="chat-card" aria-label="Admissions chat">
        <header><div className="crest">JGI</div><div><strong>JHS Admissions Assistant</strong><span>Answers from approved school information</span></div></header>
        <div className="messages" aria-live="polite">
          {messages.map((message, index) => (
            <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
              <p>{message.text}</p>
              {message.result?.citations.map((citation) => (
                <small key={citation.knowledge_id}>Source: {citation.source_label}</small>
              ))}
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

        {latestEscalation?.outcome === "escalated" && (
          <form className="contact-form" onSubmit={shareEmail}>
            <label htmlFor="email">Email for admissions follow-up</label>
            <div><input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required /><button>Send</button></div>
            <small>By submitting, you consent to being contacted about this question.</small>
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
