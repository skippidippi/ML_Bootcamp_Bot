import psycopg2

from psycopg2.extensions import connection as _connection
from typing import List, Dict
from uuid import UUID
from .config import DB_URL


def get_db_connection() -> _connection:
    return psycopg2.connect(DB_URL)


def init_db() -> None:
    """
    Создаёт таблицу messages, если она не существует.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id UUID PRIMARY KEY,
                    dialog_id UUID NOT NULL,
                    text TEXT NOT NULL,
                    participant_index INT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
            """)
        conn.commit()
    finally:
        conn.close()


def insert_message(msg_id: UUID, dialog_id: UUID, text: str, participant_index: int) -> None:
    """
    Сохраняет сообщение в таблицу messages.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO messages (id, dialog_id, text, participant_index)
                VALUES (%s, %s, %s, %s);
                """,
                (str(msg_id), str(dialog_id), text, participant_index)
            )
        conn.commit()
    finally:
        conn.close()


def select_messages_by_dialog(dialog_id: UUID) -> List[Dict]:
    """
    Возвращает список всех сообщений (text, participant_index)
    в порядке возрастания created_at для указанного dialog_id.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT text, participant_index
                FROM messages
                WHERE dialog_id = %s
                ORDER BY created_at ASC;
                """,
                (str(dialog_id),)
            )
            rows = cur.fetchall()
        return [
            {"text": row[0], "participant_index": row[1]}
            for row in rows
        ]
    finally:
        conn.close()
