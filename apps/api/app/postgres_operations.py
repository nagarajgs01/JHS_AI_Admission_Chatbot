from uuid import UUID

from .database import database_connection
from .models import (
    AdminQuestionResponse,
    AdminUnansweredQuestion,
    Lead,
    LeadSubmission,
    UnansweredQuestion,
)


def school_uuid(cursor, school_slug: str) -> UUID:
    cursor.execute("SELECT id FROM schools WHERE slug = %s", (school_slug,))
    row = cursor.fetchone()
    if row is None:
        raise LookupError(f"Unknown school: {school_slug}")
    return row[0]


class PostgresUnansweredRepository:
    def create(self, school_id: str, question: str) -> UnansweredQuestion:
        with database_connection() as connection:
            with connection.cursor() as cursor:
                database_school_id = school_uuid(cursor, school_id)
                cursor.execute(
                    """
                    INSERT INTO unanswered_questions (school_id, question)
                    VALUES (%s, %s)
                    RETURNING id, status, created_at
                    """,
                    (database_school_id, question),
                )
                unanswered_id, status, created_at = cursor.fetchone()
            connection.commit()

        return UnansweredQuestion(
            id=unanswered_id,
            school_id=school_id,
            question=question,
            status=status,
            created_at=created_at,
        )

    def list_for_school(
        self,
        school_id: str,
        question_status: str = "open",
        limit: int = 50,
        offset: int = 0,
    ) -> list[AdminUnansweredQuestion]:
        with database_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        uq.id, s.slug, uq.question, uq.contact_email,
                        uq.consent_to_contact, uq.status, uq.created_at,
                        latest.id, latest.answer, latest.status, latest.created_at,
                        latest.delivery_status, latest.delivery_error, latest.delivered_at
                    FROM unanswered_questions uq
                    JOIN schools s ON s.id = uq.school_id
                    LEFT JOIN LATERAL (
                        SELECT qr.id, qr.answer, qr.status, qr.created_at,
                               qr.delivery_status, qr.delivery_error, qr.delivered_at
                        FROM question_responses qr
                        WHERE qr.unanswered_question_id = uq.id
                        ORDER BY qr.created_at DESC
                        LIMIT 1
                    ) latest ON true
                    WHERE s.slug = %s AND uq.status = %s
                    ORDER BY uq.created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (school_id, question_status, limit, offset),
                )
                rows = cursor.fetchall()

        return [self._admin_question_from_row(row) for row in rows]

    def get_for_school(
        self,
        school_id: str,
        unanswered_id: UUID,
    ) -> AdminUnansweredQuestion | None:
        with database_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        uq.id, s.slug, uq.question, uq.contact_email,
                        uq.consent_to_contact, uq.status, uq.created_at,
                        latest.id, latest.answer, latest.status, latest.created_at,
                        latest.delivery_status, latest.delivery_error, latest.delivered_at
                    FROM unanswered_questions uq
                    JOIN schools s ON s.id = uq.school_id
                    LEFT JOIN LATERAL (
                        SELECT qr.id, qr.answer, qr.status, qr.created_at,
                               qr.delivery_status, qr.delivery_error, qr.delivered_at
                        FROM question_responses qr
                        WHERE qr.unanswered_question_id = uq.id
                        ORDER BY qr.created_at DESC
                        LIMIT 1
                    ) latest ON true
                    WHERE s.slug = %s AND uq.id = %s
                    """,
                    (school_id, unanswered_id),
                )
                row = cursor.fetchone()

        return self._admin_question_from_row(row) if row else None

    def save_answer(
        self,
        school_id: str,
        unanswered_id: UUID,
        answer: str,
        response_status: str,
        delivery_status: str = "not_applicable",
    ) -> AdminQuestionResponse:
        with database_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT uq.id
                    FROM unanswered_questions uq
                    JOIN schools s ON s.id = uq.school_id
                    WHERE uq.id = %s AND s.slug = %s
                    FOR UPDATE OF uq
                    """,
                    (unanswered_id, school_id),
                )
                if cursor.fetchone() is None:
                    raise LookupError("Unanswered question not found")

                cursor.execute(
                    """
                    INSERT INTO question_responses (
                        unanswered_question_id, answer, status, approved_at,
                        delivery_status
                    )
                    VALUES (
                        %s, %s, %s,
                        CASE WHEN %s = 'approved' THEN now() ELSE NULL END,
                        %s
                    )
                    RETURNING id, created_at, approved_at, delivery_status
                    """,
                    (
                        unanswered_id,
                        answer.strip(),
                        response_status,
                        response_status,
                        delivery_status,
                    ),
                )
                response_id, created_at, approved_at, saved_delivery_status = cursor.fetchone()

                if response_status == "approved":
                    cursor.execute(
                        """
                        UPDATE unanswered_questions
                        SET status = 'answered', resolved_at = now()
                        WHERE id = %s
                        """,
                        (unanswered_id,),
                    )
            connection.commit()

        return AdminQuestionResponse(
            id=response_id,
            unanswered_id=unanswered_id,
            answer=answer.strip(),
            status=response_status,
            created_at=created_at,
            approved_at=approved_at,
            delivery_status=saved_delivery_status,
        )

    def update_delivery(
        self,
        response_id: UUID,
        delivery_status: str,
        delivery_error: str | None = None,
    ) -> None:
        with database_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE question_responses
                    SET delivery_status = %s,
                        delivery_error = %s,
                        delivered_at = CASE WHEN %s = 'sent' THEN now() ELSE NULL END
                    WHERE id = %s AND status = 'approved'
                    """,
                    (delivery_status, delivery_error, delivery_status, response_id),
                )
                if cursor.rowcount != 1:
                    raise LookupError("Approved response not found")
            connection.commit()

    @staticmethod
    def _admin_question_from_row(row) -> AdminUnansweredQuestion:
        return AdminUnansweredQuestion(
            id=row[0],
            school_id=row[1],
            question=row[2],
            contact_email=row[3],
            consent_to_contact=row[4],
            status=row[5],
            created_at=row[6],
            latest_response_id=row[7],
            latest_answer=row[8],
            response_status=row[9],
            responded_at=row[10],
            delivery_status=row[11],
            delivery_error=row[12],
            delivered_at=row[13],
        )

    def get(self, unanswered_id: UUID) -> UnansweredQuestion | None:
        with database_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        uq.id, s.slug, uq.question, uq.contact_email,
                        uq.consent_to_contact, uq.status, uq.created_at
                    FROM unanswered_questions uq
                    JOIN schools s ON s.id = uq.school_id
                    WHERE uq.id = %s
                    """,
                    (unanswered_id,),
                )
                row = cursor.fetchone()

        if row is None:
            return None
        return UnansweredQuestion(
            id=row[0],
            school_id=row[1],
            question=row[2],
            contact_email=row[3],
            consent_to_contact=row[4],
            status=row[5],
            created_at=row[6],
        )

    def attach_contact(self, unanswered_id: UUID, email: str) -> None:
        with database_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE unanswered_questions
                    SET contact_email = %s, consent_to_contact = true
                    WHERE id = %s
                    """,
                    (email, unanswered_id),
                )
                if cursor.rowcount != 1:
                    raise LookupError("Unanswered question not found")
            connection.commit()

    def link_resolution(
        self,
        school_id: str,
        unanswered_id: UUID,
        knowledge_entry_id: UUID,
    ) -> None:
        with database_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE unanswered_questions uq
                    SET resolution_knowledge_entry_id = %s
                    FROM schools s
                    WHERE uq.id = %s
                      AND uq.school_id = s.id
                      AND s.slug = %s
                    """,
                    (knowledge_entry_id, unanswered_id, school_id),
                )
                if cursor.rowcount != 1:
                    raise LookupError("Unanswered question not found")
            connection.commit()


class PostgresLeadRepository:
    def create(self, submission: LeadSubmission) -> Lead:
        with database_connection() as connection:
            with connection.cursor() as cursor:
                database_school_id = school_uuid(cursor, submission.school_id)
                cursor.execute(
                    """
                    INSERT INTO admission_leads (
                        school_id, student_name, grade_applied_for,
                        guardian_name, mobile_number, email,
                        preferred_contact, consent_to_contact
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, status, created_at
                    """,
                    (
                        database_school_id,
                        submission.student_name,
                        submission.grade_applied_for,
                        submission.guardian_name,
                        submission.mobile_number,
                        submission.email,
                        submission.preferred_contact,
                        submission.consent_to_contact,
                    ),
                )
                lead_id, status, created_at = cursor.fetchone()
            connection.commit()

        return Lead(
            **submission.model_dump(),
            id=lead_id,
            status=status,
            created_at=created_at,
        )
