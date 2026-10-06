export type Citation = {
  knowledge_id: string;
  title: string;
  source_label: string;
  excerpt: string;
  score: number;
};

export type ChatResponse = {
  conversation_id: string;
  answer: string;
  outcome: "answered" | "escalated";
  confidence: number;
  citations: Citation[];
  unanswered_id?: string;
};

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export async function sendQuestion(message: string, conversationId?: string): Promise<ChatResponse> {
  const response = await fetch(`${API_URL}/v1/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      school_id: "jhs-electronic-city",
      message,
      conversation_id: conversationId,
    }),
  });
  if (!response.ok) throw new Error("The assistant is temporarily unavailable.");
  return response.json();
}

export async function submitContact(unansweredId: string, email: string): Promise<void> {
  const response = await fetch(`${API_URL}/v1/escalations/contact`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      school_id: "jhs-electronic-city",
      unanswered_id: unansweredId,
      email,
      consent_to_contact: true,
    }),
  });
  if (!response.ok) throw new Error("We could not save your contact information.");
}

export type LeadInput = {
  student_name: string;
  grade_applied_for: string;
  guardian_name: string;
  mobile_number: string;
  email: string;
  preferred_contact?: string;
  consent_to_contact: boolean;
};

export async function submitLead(lead: LeadInput): Promise<void> {
  const response = await fetch(`${API_URL}/v1/leads`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ school_id: "jhs-electronic-city", ...lead }),
  });
  if (!response.ok) throw new Error("We could not submit your enquiry.");
}
