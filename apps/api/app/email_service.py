import asyncio
import smtplib
from email.message import EmailMessage
from typing import Protocol

from .config import Settings


class EmailSender(Protocol):
    enabled: bool

    async def send_parent_answer(
        self,
        recipient: str,
        question: str,
        answer: str,
    ) -> None: ...


class DisabledEmailSender:
    enabled = False

    async def send_parent_answer(self, recipient: str, question: str, answer: str) -> None:
        del recipient, question, answer
        raise RuntimeError("Email delivery is not configured")


class SMTPEmailSender:
    enabled = True

    def __init__(self, settings: Settings) -> None:
        self.host = settings.smtp_host
        self.port = settings.smtp_port
        self.username = settings.smtp_username
        self.password = settings.smtp_password
        self.use_tls = settings.smtp_use_tls
        self.sender = settings.email_from

    def _send(self, recipient: str, question: str, answer: str) -> None:
        message = EmailMessage()
        message["From"] = self.sender
        message["To"] = recipient
        message["Subject"] = "Answer to your Jain Heritage School enquiry"
        message.set_content(
            "Hello,\n\n"
            "The Jain Heritage School admissions team has answered your enquiry.\n\n"
            f"Your question:\n{question.strip()}\n\n"
            f"School response:\n{answer.strip()}\n\n"
            "Regards,\nJain Heritage School Electronic City Admissions"
        )
        with smtplib.SMTP(self.host, self.port, timeout=15) as client:
            if self.use_tls:
                client.starttls()
            if self.username:
                client.login(self.username, self.password or "")
            client.send_message(message)

    async def send_parent_answer(self, recipient: str, question: str, answer: str) -> None:
        await asyncio.to_thread(self._send, recipient, question, answer)


def build_email_sender(settings: Settings) -> EmailSender:
    if settings.email_provider.lower() == "smtp":
        return SMTPEmailSender(settings)
    return DisabledEmailSender()
