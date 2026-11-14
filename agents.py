import re
import time

import database as db
import llm

POSITIVE = "positive_feedback"
NEGATIVE = "negative_feedback"
QUERY = "query"

CATEGORY_LABELS = {
    POSITIVE: "Positive Feedback",
    NEGATIVE: "Negative Feedback",
    QUERY: "Query",
}

TICKET_PATTERN = re.compile(r"(?<!\d)(\d{6})(?!\d)")

POSITIVE_WORDS = [
    "thank", "thanks", "appreciate", "great", "excellent", "awesome", "happy",
    "love", "helpful", "wonderful", "amazing", "good job", "well done", "fantastic",
    "sorted", "resolved quickly", "pleased", "grateful", "kudos", "smooth",
]

NEGATIVE_WORDS = [
    "not ", "n't", "never", "still", "problem", "issue", "bad", "worst", "terrible",
    "delay", "delayed", "failed", "failing", "unable", "disappointed", "angry",
    "frustrated", "wrong", "error", "charged twice", "double charge", "blocked",
    "poor", "slow", "complaint", "lost", "missing", "unhappy", "rude", "waiting",
    "pending", " but ", "haven't received", "no response",
]

QUERY_WORDS = [
    "status", "check", "update on", "track", "what is", "what's", "where is",
    "any update", "progress", "how long", "when will",
]

CLASSIFIER_PROMPT = """You are the Classifier Agent for a bank's customer support desk.
Read the customer's message and decide which ONE category it belongs to:

- positive_feedback: the customer is happy, thankful or praising the bank.
- negative_feedback: the customer is unhappy, complaining, or reporting a problem that is not yet fixed.
- query: the customer is asking a question, especially about the status of a support ticket.

Rules:
- If the message asks about a ticket's status or contains a ticket number with a question, it is a query.
- Mixed messages like "thanks, but my card still has not arrived" are negative_feedback.

Reply with JSON only, using these keys:
{"category": "positive_feedback" | "negative_feedback" | "query",
 "customer_name": string or null,
 "ticket_number": six digit string or null,
 "confidence": number between 0 and 1,
 "reason": short explanation}"""

POSITIVE_PROMPT = """You are the Feedback Handler Agent for a friendly bank.
Write a warm, personalised thank-you reply to a happy customer.
The reply MUST start with exactly this sentence: "{opening}"
After that, add at most one short, sincere sentence that refers to what the customer thanked us for.
Keep it under 45 words. No emojis, no signatures, no ticket numbers."""


def positive_opening(name):
    if name:
        return f"Thank you for your kind words, {name}! We're delighted to assist you."
    return "Thank you for your kind words! We're delighted to assist you."


def extract_ticket_number(message):
    match = TICKET_PATTERN.search(message or "")
    return match.group(1) if match else None


def rule_based_category(message):
    text = f" {message.lower()} "
    has_ticket = extract_ticket_number(message) is not None
    is_question = "?" in text or any(w in text for w in QUERY_WORDS)
    negative_hits = sum(1 for w in NEGATIVE_WORDS if w in text)
    positive_hits = sum(1 for w in POSITIVE_WORDS if w in text)

    if has_ticket and (is_question or "ticket" in text):
        return QUERY
    if is_question and "ticket" in text:
        return QUERY
    if negative_hits and negative_hits >= positive_hits:
        return NEGATIVE
    if positive_hits:
        return POSITIVE
    if is_question:
        return QUERY
    return NEGATIVE


class ClassifierAgent:
    name = "Classifier Agent"

    def run(self, message):
        data, trace = llm.chat(CLASSIFIER_PROMPT, message, json_mode=True, temperature=0)
        trace["agent"] = self.name
        if data and data.get("category") in CATEGORY_LABELS:
            return {
                "category": data["category"],
                "customer_name": data.get("customer_name") or None,
                "ticket_number": data.get("ticket_number") or None,
                "confidence": data.get("confidence"),
                "reason": data.get("reason", ""),
                "method": "LLM",
            }, trace
        category = rule_based_category(message)
        trace["fallback"] = "Rule-based keyword classifier used"
        return {
            "category": category,
            "customer_name": None,
            "ticket_number": extract_ticket_number(message),
            "confidence": None,
            "reason": "Keyword rules (LLM unavailable or returned an invalid answer)",
            "method": "Rules",
        }, trace


class FeedbackHandlerAgent:
    name = "Feedback Handler Agent"

    def handle_positive(self, message, customer_name):
        opening = positive_opening(customer_name)
        text, trace = llm.chat(
            POSITIVE_PROMPT.format(opening=opening),
            f"Customer message: {message}",
            temperature=0.7,
        )
        trace["agent"] = f"{self.name} (positive)"
        if text and text.startswith(opening):
            response = text
        else:
            response = opening
            trace["fallback"] = "Template thank-you message used"
        return {
            "response": response,
            "ticket_id": None,
            "ticket_action": None,
            "db_operation": "No database change needed",
        }, trace

    def handle_negative(self, message, customer_name):
        ticket_id = db.create_ticket(customer_name, message)
        response = (
            f"We apologize for the inconvenience. A new ticket #{ticket_id} has been generated, "
            "and our team will follow up shortly."
        )
        trace = {
            "agent": f"{self.name} (negative)",
            "action": "INSERT INTO support_tickets",
            "ticket_id": ticket_id,
            "status": "Open",
        }
        return {
            "response": response,
            "ticket_id": ticket_id,
            "ticket_action": "created",
            "db_operation": f"INSERT support_tickets (ticket_id={ticket_id}, status='Open')",
        }, trace


class QueryHandlerAgent:
    name = "Query Handler Agent"

    def run(self, message, hinted_ticket=None):
        ticket_id = extract_ticket_number(message) or (
            hinted_ticket if hinted_ticket and TICKET_PATTERN.fullmatch(str(hinted_ticket)) else None
        )
        trace = {"agent": self.name, "extracted_ticket": ticket_id}
        if not ticket_id:
            trace["action"] = "No ticket number found"
            return {
                "response": "I'd be glad to check that for you. Could you please share your 6-digit ticket number?",
                "ticket_id": None,
                "ticket_action": "missing_ticket_number",
                "db_operation": "No lookup (ticket number missing)",
            }, trace

        ticket = db.get_ticket(ticket_id)
        trace["action"] = f"SELECT status FROM support_tickets WHERE ticket_id='{ticket_id}'"
        if not ticket:
            trace["result"] = "not found"
            return {
                "response": f"Sorry, I couldn't find ticket #{ticket_id}. Please check the number and try again.",
                "ticket_id": ticket_id,
                "ticket_action": "not_found",
                "db_operation": f"SELECT support_tickets WHERE ticket_id={ticket_id} -> no record",
            }, trace

        trace["result"] = ticket["status"]
        return {
            "response": f"Your ticket #{ticket_id} is currently marked as: {ticket['status']}.",
            "ticket_id": ticket_id,
            "ticket_action": "status_lookup",
            "db_operation": f"SELECT support_tickets WHERE ticket_id={ticket_id} -> {ticket['status']}",
        }, trace


class SupportOrchestrator:
    def __init__(self):
        self.classifier = ClassifierAgent()
        self.feedback = FeedbackHandlerAgent()
        self.query = QueryHandlerAgent()

    def process(self, message, customer_name=None, source="chat"):
        started = time.time()
        customer_name = (customer_name or "").strip() or None
        record = {
            "source": source,
            "message": message,
            "customer_name": customer_name,
            "agent_path": [self.classifier.name],
            "traces": [],
            "success": True,
            "error": None,
        }
        try:
            classification, trace = self.classifier.run(message)
            record["traces"].append(trace)
            category = classification["category"]
            name = customer_name or classification.get("customer_name")
            record.update(
                {
                    "category": category,
                    "category_label": CATEGORY_LABELS[category],
                    "classifier_method": classification["method"],
                    "confidence": classification.get("confidence"),
                    "reason": classification.get("reason"),
                    "customer_name": name,
                }
            )

            if category == POSITIVE:
                record["agent_path"].append("Feedback Handler (Positive)")
                outcome, trace = self.feedback.handle_positive(message, name)
            elif category == NEGATIVE:
                record["agent_path"].append("Feedback Handler (Negative)")
                outcome, trace = self.feedback.handle_negative(message, name)
            else:
                record["agent_path"].append(self.query.name)
                outcome, trace = self.query.run(message, classification.get("ticket_number"))

            record["traces"].append(trace)
            record.update(outcome)
        except Exception as exc:
            record["success"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["response"] = "Sorry, something went wrong while handling your message. Please try again."
            record.setdefault("category", None)
            record.setdefault("category_label", "Unknown")
            record.setdefault("db_operation", "None")

        record["latency_ms"] = int((time.time() - started) * 1000)
        record["log_id"] = db.log_interaction(record)
        return record
