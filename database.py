import json
import os
import random
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta

DB_PATH = os.getenv("SUPPORT_DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "support.db"))

TICKET_STATUSES = ["Open", "In Progress", "Resolved", "Closed"]

SEED_TICKETS = [
    ("650932", "Ravi Kumar", "Net banking login keeps failing", "Resolved", 9),
    ("123456", "Anita Sharma", "Credit card statement shows a double charge", "In Progress", 4),
    ("456789", "John Mathew", "Cheque book request not delivered", "Open", 2),
    ("112233", "Priya Nair", "UPI payment debited but not credited", "Resolved", 12),
    ("998877", "Sameer Khan", "Unable to update mobile number", "Closed", 20),
]


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS support_tickets (
                ticket_id TEXT PRIMARY KEY,
                customer_name TEXT,
                issue TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Open',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS interaction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'chat',
                customer_name TEXT,
                message TEXT NOT NULL,
                category TEXT,
                classifier_method TEXT,
                agent_path TEXT,
                response TEXT,
                ticket_id TEXT,
                ticket_action TEXT,
                success INTEGER NOT NULL DEFAULT 1,
                error TEXT,
                latency_ms INTEGER,
                prompt_trace TEXT,
                user_rating INTEGER
            );

            CREATE TABLE IF NOT EXISTS evaluation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                total_cases INTEGER,
                classification_accuracy REAL,
                routing_success_rate REAL,
                response_pass_rate REAL,
                avg_empathy REAL,
                avg_clarity REAL,
                details TEXT
            );
            """
        )
        count = conn.execute("SELECT COUNT(*) FROM support_tickets").fetchone()[0]
        if count == 0:
            for ticket_id, name, issue, status, days_ago in SEED_TICKETS:
                stamp = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")
                conn.execute(
                    "INSERT INTO support_tickets VALUES (?, ?, ?, ?, ?, ?)",
                    (ticket_id, name, issue, status, stamp, stamp),
                )


def generate_ticket_id():
    with get_conn() as conn:
        while True:
            candidate = str(random.randint(100000, 999999))
            exists = conn.execute(
                "SELECT 1 FROM support_tickets WHERE ticket_id = ?", (candidate,)
            ).fetchone()
            if not exists:
                return candidate


def create_ticket(customer_name, issue):
    ticket_id = generate_ticket_id()
    stamp = now()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO support_tickets VALUES (?, ?, ?, 'Open', ?, ?)",
            (ticket_id, customer_name or "Customer", issue, stamp, stamp),
        )
    return ticket_id


def get_ticket(ticket_id):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM support_tickets WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
    return dict(row) if row else None


def update_ticket_status(ticket_id, status):
    with get_conn() as conn:
        conn.execute(
            "UPDATE support_tickets SET status = ?, updated_at = ? WHERE ticket_id = ?",
            (status, now(), ticket_id),
        )


def delete_ticket(ticket_id):
    with get_conn() as conn:
        conn.execute("DELETE FROM support_tickets WHERE ticket_id = ?", (ticket_id,))


def list_tickets():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM support_tickets ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def log_interaction(record):
    with get_conn() as conn:
        cursor = conn.execute(
            """
            INSERT INTO interaction_logs (
                created_at, source, customer_name, message, category, classifier_method,
                agent_path, response, ticket_id, ticket_action, success, error,
                latency_ms, prompt_trace
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now(),
                record.get("source", "chat"),
                record.get("customer_name"),
                record.get("message"),
                record.get("category"),
                record.get("classifier_method"),
                " -> ".join(record.get("agent_path", [])),
                record.get("response"),
                record.get("ticket_id"),
                record.get("ticket_action"),
                1 if record.get("success", True) else 0,
                record.get("error"),
                record.get("latency_ms"),
                json.dumps(record.get("traces", []), indent=2),
            ),
        )
        return cursor.lastrowid


def rate_interaction(log_id, rating):
    with get_conn() as conn:
        conn.execute(
            "UPDATE interaction_logs SET user_rating = ? WHERE id = ?", (rating, log_id)
        )


def list_logs(limit=500):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM interaction_logs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def clear_logs():
    with get_conn() as conn:
        conn.execute("DELETE FROM interaction_logs")


def save_evaluation_run(summary, details):
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO evaluation_runs (
                created_at, total_cases, classification_accuracy, routing_success_rate,
                response_pass_rate, avg_empathy, avg_clarity, details
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now(),
                summary["total_cases"],
                summary["classification_accuracy"],
                summary["routing_success_rate"],
                summary["response_pass_rate"],
                summary["avg_empathy"],
                summary["avg_clarity"],
                json.dumps(details),
            ),
        )


def list_evaluation_runs():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM evaluation_runs ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]
