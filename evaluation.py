import re

import database as db
import llm
from agents import NEGATIVE, POSITIVE, QUERY, SupportOrchestrator, positive_opening

TEST_CASES = [
    {"message": "Thanks for sorting out my net banking login issue.", "name": "Ravi", "expected": POSITIVE},
    {"message": "Thanks for resolving my credit card issue so quickly.", "name": "Anita", "expected": POSITIVE},
    {"message": "The staff at your Andheri branch were very helpful today. Great service!", "name": "", "expected": POSITIVE},
    {"message": "I really appreciate how fast my loan was approved.", "name": "Meera", "expected": POSITIVE},
    {"message": "Your new mobile app is excellent, transfers are so smooth now.", "name": "", "expected": POSITIVE},
    {"message": "My debit card replacement still hasn't arrived.", "name": "John", "expected": NEGATIVE},
    {"message": "I was charged twice for the same transaction and nobody is helping me.", "name": "Sameer", "expected": NEGATIVE},
    {"message": "The ATM swallowed my card and I am very frustrated.", "name": "", "expected": NEGATIVE},
    {"message": "Net banking has been down since morning, this is terrible.", "name": "Priya", "expected": NEGATIVE},
    {"message": "Thanks, but my cheque book request is still pending after three weeks.", "name": "", "expected": NEGATIVE},
    {"message": "Could you check the status of ticket 650932?", "name": "", "expected": QUERY},
    {"message": "What is the status of my ticket 123456?", "name": "Anita", "expected": QUERY},
    {"message": "Any update on ticket #456789?", "name": "", "expected": QUERY},
    {"message": "Can you tell me where my complaint 998877 stands?", "name": "Sameer", "expected": QUERY},
    {"message": "Please check ticket 000111 for me.", "name": "", "expected": QUERY},
    {"message": "What's the status of my support ticket?", "name": "", "expected": QUERY},
]

EXPECTED_ROUTE = {
    POSITIVE: "Feedback Handler (Positive)",
    NEGATIVE: "Feedback Handler (Negative)",
    QUERY: "Query Handler Agent",
}

JUDGE_PROMPT = """You are a strict quality reviewer for a bank's customer support replies.
Score the support reply on two scales from 1 (poor) to 5 (excellent):
- empathy: does it sound warm, caring and appropriate for the customer's mood?
- clarity: is it clear, short and does it give the customer exactly the information they need?
Reply in JSON: {"empathy": int, "clarity": int, "comment": short string}"""


def check_response(case, result):
    response = result.get("response") or ""
    category = result.get("category")
    if category == POSITIVE:
        name = result.get("customer_name")
        ok = response.startswith(positive_opening(name)) or response.startswith("Thank you for your kind words")
        return ok, "Starts with the thank-you format" if ok else "Thank-you format missing"
    if category == NEGATIVE:
        match = re.search(r"ticket #(\d{6})", response)
        if not match:
            return False, "No 6-digit ticket number in reply"
        ticket = db.get_ticket(match.group(1))
        if not ticket or ticket["status"] != "Open":
            return False, "Ticket not saved as Open in database"
        return True, f"Ticket #{match.group(1)} saved as Open"
    if category == QUERY:
        ticket_id = result.get("ticket_id")
        if not ticket_id:
            ok = "ticket number" in response.lower()
            return ok, "Asked for the ticket number" if ok else "Did not ask for a ticket number"
        ticket = db.get_ticket(ticket_id)
        if ticket:
            expected = f"Your ticket #{ticket_id} is currently marked as: {ticket['status']}."
            ok = response == expected
            return ok, "Status matches database" if ok else "Status reply does not match database"
        ok = "couldn't find" in response.lower()
        return ok, "Unknown ticket handled politely" if ok else "Unknown ticket not handled"
    return False, "No category returned"


def heuristic_scores(result):
    response = (result.get("response") or "").lower()
    warm_words = ["thank", "delighted", "apolog", "sorry", "glad", "happy", "please"]
    empathy = min(5, 2 + sum(1 for w in warm_words if w in response))
    clarity = 5 if len(response.split()) <= 45 else 3
    return empathy, clarity, "Heuristic score (no LLM key)"


def judge(case, result):
    if not llm.llm_available():
        return heuristic_scores(result)
    data, _ = llm.chat(
        JUDGE_PROMPT,
        f"Customer message: {case['message']}\nSupport reply: {result.get('response')}",
        json_mode=True,
        temperature=0,
    )
    try:
        return int(data["empathy"]), int(data["clarity"]), data.get("comment", "")
    except Exception:
        return heuristic_scores(result)


def run_evaluation(cases=None, use_judge=True, progress=None):
    cases = cases or TEST_CASES
    orchestrator = SupportOrchestrator()
    rows = []
    created_tickets = []

    for index, case in enumerate(cases):
        result = orchestrator.process(case["message"], case.get("name"), source="evaluation")
        predicted = result.get("category")
        classified_ok = predicted == case["expected"]
        route = (result.get("agent_path") or [None])[-1]
        routed_ok = result.get("success", False) and route == EXPECTED_ROUTE.get(predicted)
        response_ok, response_note = check_response(case, result)
        if use_judge:
            empathy, clarity, comment = judge(case, result)
        else:
            empathy, clarity, comment = heuristic_scores(result)
        if result.get("ticket_action") == "created":
            created_tickets.append(result["ticket_id"])

        rows.append(
            {
                "message": case["message"],
                "expected": case["expected"],
                "predicted": predicted,
                "classifier": result.get("classifier_method"),
                "classification_correct": classified_ok,
                "routed_correctly": routed_ok,
                "response_check": response_ok,
                "check_note": response_note,
                "empathy": empathy,
                "clarity": clarity,
                "judge_comment": comment,
                "response": result.get("response"),
                "latency_ms": result.get("latency_ms"),
            }
        )
        if progress:
            progress((index + 1) / len(cases), case["message"])

    for ticket_id in created_tickets:
        db.delete_ticket(ticket_id)

    total = len(rows)
    summary = {
        "total_cases": total,
        "classification_accuracy": round(100 * sum(r["classification_correct"] for r in rows) / total, 1),
        "routing_success_rate": round(100 * sum(r["routed_correctly"] for r in rows) / total, 1),
        "response_pass_rate": round(100 * sum(r["response_check"] for r in rows) / total, 1),
        "avg_empathy": round(sum(r["empathy"] for r in rows) / total, 2),
        "avg_clarity": round(sum(r["clarity"] for r in rows) / total, 2),
    }
    per_category = {}
    for category in (POSITIVE, NEGATIVE, QUERY):
        subset = [r for r in rows if r["expected"] == category]
        if subset:
            per_category[category] = round(100 * sum(r["classification_correct"] for r in subset) / len(subset), 1)
    summary["per_category_accuracy"] = per_category

    db.save_evaluation_run(summary, rows)
    return summary, rows
