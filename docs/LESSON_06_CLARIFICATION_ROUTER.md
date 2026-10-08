# Lesson 06: Answer, Clarify, or Escalate

The public chat router now has four outcomes. Strong evidence produces an answer.
Ambiguous or misspelled input with related approved topics produces clickable
clarification questions. A clear request for missing, current, or personal information
still creates an escalation. An obvious non-school request receives a scope reminder
without creating work for the admissions team.

Clarification candidate retrieval is deliberately separate from answer retrieval.
Loose similarity is allowed to identify possible topics, but never to supply facts.
Every loose candidate must also share an intent term with its approved entry title or
tags after query expansion. This prevents generic school similarity from suggesting
transport, age, or assessment for a clear school-timings question.
Clicking a suggested question sends a new chat request through the complete strict RAG
and grounding-verification pipeline.

The `None of these` action resubmits the original wording with `force_escalation=true`.
This creates the unanswered record only after the parent rejects the proposed intents,
avoiding unnecessary work for school staff without hiding unresolved questions.

Obvious non-school requests, such as the current time or weather, use an
`out_of_scope` response. They do not create admissions tickets. Domain expansion also
maps informal phrases such as "study options" to approved curriculum and board topics.
