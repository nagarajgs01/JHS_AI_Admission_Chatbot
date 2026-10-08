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
  outcome: "answered" | "clarification" | "escalated" | "out_of_scope";
  confidence: number;
  citations: Citation[];
  unanswered_id?: string;
  suggested_questions: string[];
  clarification_question?: string;
  clarification_kind?: "choice" | "details";
  required_details: string[];
};

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export async function sendQuestion(
  message: string,
  conversationId?: string,
  forceEscalation = false,
  context: Record<string, string> = {},
): Promise<ChatResponse> {
  const response = await fetch(`${API_URL}/v1/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      school_id: "jhs-electronic-city",
      message,
      conversation_id: conversationId,
      force_escalation: forceEscalation,
      context,
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

export type AdminQuestion = {
  id: string;
  school_id: string;
  question: string;
  contact_email?: string;
  consent_to_contact: boolean;
  status: "open" | "answered" | "closed";
  created_at: string;
  latest_answer?: string;
  response_status?: "draft" | "approved";
  responded_at?: string;
};

export type AdminSuggestion = {
  answer: string;
  kind: "grounded" | "template" | "callback";
  requires_staff_verification: boolean;
  citations: Citation[];
};

export type AdminSuggestionResponse = {
  unanswered_id: string;
  question: string;
  suggestions: AdminSuggestion[];
};

export type AdminQuestionResponse = {
  id: string;
  unanswered_id: string;
  answer: string;
  status: "draft" | "approved";
  created_at: string;
  approved_at?: string;
  knowledge_entry_id?: string;
};

export type AdminApprovalOptions = {
  publish_to_knowledge: boolean;
  knowledge_title?: string;
  valid_from?: string;
  expires_at?: string;
  supersede_knowledge_ids?: string[];
};

async function adminRequest<T>(path: string, adminKey: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      "X-Admin-Key": adminKey,
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
    },
  });
  if (!response.ok) {
    let message = "The admin request could not be completed.";
    try {
      const payload = await response.json();
      if (typeof payload.detail === "string") message = payload.detail;
    } catch {
      // Keep the safe generic message when the server did not return JSON.
    }
    const error = new Error(message);
    error.name = response.status === 401 ? "UnauthorizedError" : "AdminApiError";
    throw error;
  }
  return response.json();
}

export function listAdminQuestions(
  adminKey: string,
  questionStatus: AdminQuestion["status"],
): Promise<AdminQuestion[]> {
  const query = new URLSearchParams({
    school_id: "jhs-electronic-city",
    status: questionStatus,
  });
  return adminRequest(`/v1/admin/unanswered?${query}`, adminKey);
}

export function getAdminQuestion(adminKey: string, unansweredId: string): Promise<AdminQuestion> {
  const query = new URLSearchParams({ school_id: "jhs-electronic-city" });
  return adminRequest(`/v1/admin/unanswered/${unansweredId}?${query}`, adminKey);
}

export function generateAdminSuggestions(
  adminKey: string,
  unansweredId: string,
): Promise<AdminSuggestionResponse> {
  const query = new URLSearchParams({ school_id: "jhs-electronic-city" });
  return adminRequest(`/v1/admin/unanswered/${unansweredId}/suggestions?${query}`, adminKey, {
    method: "POST",
  });
}

function submitAdminAnswer(
  adminKey: string,
  unansweredId: string,
  answer: string,
  action: "draft" | "approve",
  options: Partial<AdminApprovalOptions> = {},
): Promise<AdminQuestionResponse> {
  return adminRequest(`/v1/admin/unanswered/${unansweredId}/${action}`, adminKey, {
    method: "POST",
    body: JSON.stringify({ school_id: "jhs-electronic-city", answer, ...options }),
  });
}

export function saveAdminDraft(adminKey: string, unansweredId: string, answer: string) {
  return submitAdminAnswer(adminKey, unansweredId, answer, "draft");
}

export function approveAdminAnswer(
  adminKey: string,
  unansweredId: string,
  answer: string,
  options: AdminApprovalOptions,
) {
  return submitAdminAnswer(adminKey, unansweredId, answer, "approve", options);
}
