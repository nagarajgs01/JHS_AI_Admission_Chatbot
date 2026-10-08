import asyncio

from app.config import Settings
from app.email_service import SMTPEmailSender, build_email_sender


class FakeSMTP:
    last_instance = None

    def __init__(self, host, port, timeout):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.started_tls = False
        self.login_values = None
        self.message = None
        FakeSMTP.last_instance = self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def starttls(self):
        self.started_tls = True

    def login(self, username, password):
        self.login_values = (username, password)

    def send_message(self, message):
        self.message = message


def test_smtp_sender_builds_parent_answer_email(monkeypatch):
    monkeypatch.setattr("app.email_service.smtplib.SMTP", FakeSMTP)
    settings = Settings(
        email_provider="smtp",
        smtp_host="mailpit",
        smtp_port=1025,
        email_from="admissions@example.edu",
    )
    sender = build_email_sender(settings)

    asyncio.run(
        sender.send_parent_answer(
            "parent@example.com",
            "Does the school provide transport?",
            "Yes, transport is available on approved routes.",
        )
    )

    smtp = FakeSMTP.last_instance
    assert isinstance(sender, SMTPEmailSender)
    assert smtp.host == "mailpit"
    assert smtp.message["To"] == "parent@example.com"
    assert "Does the school provide transport?" in smtp.message.get_content()
    assert "approved routes" in smtp.message.get_content()


def test_disabled_provider_does_not_claim_to_be_enabled():
    sender = build_email_sender(Settings(email_provider="disabled"))
    assert not sender.enabled
