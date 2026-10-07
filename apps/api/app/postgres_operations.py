from uuid import UUID

from .database import database_connection
from .models import Lead, LeadSubmission, UnansweredQuestion


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
