# Lesson 05: React Admin Response Desk

The web application now selects its interface from the URL. `/` renders the public
admissions assistant, while `/admin` renders the staff response desk.

## Credential handling

The administrator enters the server-side API key. The browser keeps it in
`sessionStorage`, sends it through the `X-Admin-Key` header, and removes it on sign-out.
This is suitable for the local baseline. Production should replace the shared key with
individual accounts, short-lived sessions, roles, and an audit identity.

## Review workflow

Staff filter open, answered, or closed questions; select a question; request safe AI
suggestions; copy a suggestion into the editable response; save a versioned draft; and
approve only after verifying it. Callback suggestions and grounded suggestions have
different visual labels, grounded suggestions display their approved sources, and
purple template suggestions provide answer-shaped blanks for missing facts. Incomplete
templates can be saved but cannot be approved.

Approval currently stores the answer and changes the question status. It explicitly
does not claim that email was sent. Reliable email delivery is the next backend stage.
