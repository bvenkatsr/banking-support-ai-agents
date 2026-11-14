import json

import pandas as pd
import streamlit as st

import database as db
import llm
from agents import CATEGORY_LABELS, NEGATIVE, POSITIVE, QUERY, SupportOrchestrator
from evaluation import TEST_CASES, run_evaluation

st.set_page_config(page_title="Banking Support AI Agents", page_icon="🏦", layout="wide")

db.init_db()

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = SupportOrchestrator()
if "chat" not in st.session_state:
    st.session_state.chat = []
if "scenario_results" not in st.session_state:
    st.session_state.scenario_results = {}

CATEGORY_COLORS = {POSITIVE: "green", NEGATIVE: "red", QUERY: "blue"}

SAMPLE_MESSAGES = [
    "Thanks for sorting out my net banking login issue.",
    "My debit card replacement still hasn't arrived.",
    "Could you check the status of ticket 650932?",
]

SCENARIOS = {
    "Positive Feedback": [
        ("Ravi", "Thanks for sorting out my net banking login issue."),
        ("Anita", "I really appreciate how quickly you fixed my credit card limit."),
    ],
    "Negative Feedback": [
        ("John", "My debit card replacement still hasn't arrived."),
        ("Sameer", "I was charged twice for my electricity bill payment."),
    ],
    "Query": [
        ("", "Could you check the status of ticket 650932?"),
        ("Anita", "What is the status of my ticket 123456?"),
        ("", "Any update on my support ticket?"),
    ],
}


def category_badge(result):
    category = result.get("category")
    label = CATEGORY_LABELS.get(category, "Unknown")
    color = CATEGORY_COLORS.get(category, "gray")
    return f":{color}-badge[{label}]"


def render_details(result):
    path = " → ".join(result.get("agent_path", []))
    cols = st.columns(4)
    cols[0].markdown(f"**Classification**  \n{category_badge(result)}")
    cols[1].markdown(f"**Classifier**  \n{result.get('classifier_method', '-')}")
    confidence = result.get("confidence")
    cols[2].markdown(f"**Confidence**  \n{confidence if confidence is not None else '-'}")
    cols[3].markdown(f"**Time**  \n{result.get('latency_ms', 0)} ms")
    st.markdown(f"**Agent path:** {path}")
    if result.get("reason"):
        st.caption(f"Why: {result['reason']}")
    st.markdown(f"**Database:** `{result.get('db_operation', 'None')}`")
    if result.get("error"):
        st.error(result["error"])
    with st.expander("Prompt trace"):
        st.json(result.get("traces", []))


def render_sidebar():
    with st.sidebar:
        st.title("🏦 Support Desk")
        if llm.llm_available():
            st.success(f"OpenAI connected · {llm.get_model()}")
        else:
            st.warning("No OpenAI key found. Running in rule-based mode.")
            st.caption("Add OPENAI_API_KEY to .env (local) or Streamlit secrets (cloud).")
        page = st.radio(
            "Go to",
            ["Support Chat", "Test Scenarios", "Tickets", "Logs & Debugging", "Evaluation"],
            label_visibility="collapsed",
        )
        st.divider()
        st.text_input("Customer name (optional)", key="customer_name", placeholder="e.g. Ravi")
        st.caption("Academic demo project. Do not enter real banking details.")
    return page


def page_chat():
    st.header("💬 Support Chat")
    st.caption("Type a message like a bank customer would. The Classifier Agent decides who should answer.")

    sample_cols = st.columns(len(SAMPLE_MESSAGES))
    pending = None
    for col, sample in zip(sample_cols, SAMPLE_MESSAGES):
        if col.button(sample, width="stretch"):
            pending = sample

    typed = st.chat_input("Write your message to the bank...")
    message = typed or pending
    if message:
        result = st.session_state.orchestrator.process(message, st.session_state.get("customer_name"))
        st.session_state.chat.append(result)

    if not st.session_state.chat:
        st.info("No messages yet. Try one of the sample buttons above.")

    for result in st.session_state.chat:
        with st.chat_message("user"):
            st.write(result["message"])
        with st.chat_message("assistant", avatar="🏦"):
            st.write(result["response"])
            with st.expander(f"Agent details · {CATEGORY_LABELS.get(result.get('category'), 'Unknown')}"):
                render_details(result)
            rating = st.feedback("thumbs", key=f"rating_{result['log_id']}")
            if rating is not None:
                db.rate_interaction(result["log_id"], rating)

    if st.session_state.chat and st.button("Clear conversation"):
        st.session_state.chat = []
        st.rerun()


def page_scenarios():
    st.header("🧪 Test Scenarios")
    st.caption("Ready-made messages for each agent role. Click one to watch the agents work.")

    if st.button("Run all scenarios", type="primary"):
        for group, items in SCENARIOS.items():
            for name, message in items:
                result = st.session_state.orchestrator.process(message, name, source="scenario")
                st.session_state.scenario_results[message] = result

    cols = st.columns(len(SCENARIOS))
    for col, (group, items) in zip(cols, SCENARIOS.items()):
        with col:
            st.subheader(group)
            for name, message in items:
                with st.container(border=True):
                    st.markdown(f"_{message}_")
                    if name:
                        st.caption(f"Customer: {name}")
                    if st.button("Run", key=f"run_{message}"):
                        result = st.session_state.orchestrator.process(message, name, source="scenario")
                        st.session_state.scenario_results[message] = result
                    result = st.session_state.scenario_results.get(message)
                    if result:
                        expected_ok = result.get("category_label") == group
                        st.markdown(f"{category_badge(result)} {'✅' if expected_ok else '⚠️'}")
                        st.write(result["response"])
                        with st.expander("Details"):
                            render_details(result)


def page_tickets():
    st.header("🎫 Support Tickets")
    tickets = db.list_tickets()
    if not tickets:
        st.info("No tickets yet.")
        return
    df = pd.DataFrame(tickets)

    counts = df["status"].value_counts()
    cols = st.columns(len(db.TICKET_STATUSES))
    for col, status in zip(cols, db.TICKET_STATUSES):
        col.metric(status, int(counts.get(status, 0)))

    status_filter = st.multiselect("Filter by status", db.TICKET_STATUSES, default=db.TICKET_STATUSES)
    st.dataframe(df[df["status"].isin(status_filter)], width="stretch", hide_index=True)

    st.subheader("Update a ticket")
    with st.form("update_ticket"):
        c1, c2 = st.columns(2)
        ticket_id = c1.selectbox("Ticket", df["ticket_id"].tolist())
        status = c2.selectbox("New status", db.TICKET_STATUSES)
        if st.form_submit_button("Save"):
            db.update_ticket_status(ticket_id, status)
            st.success(f"Ticket #{ticket_id} is now {status}.")
            st.rerun()


def page_logs():
    st.header("📜 Logs & Debugging")
    logs = db.list_logs()
    if not logs:
        st.info("No activity yet. Send a message in Support Chat first.")
        return
    df = pd.DataFrame(logs)
    df["handler"] = df["agent_path"].fillna("").str.split(" -> ").str[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Interactions", len(df))
    c2.metric("Success rate", f"{100 * df['success'].mean():.1f}%")
    c3.metric("Avg response time", f"{int(df['latency_ms'].mean())} ms")
    c4.metric("Tickets created", int((df["ticket_action"] == "created").sum()))

    left, right = st.columns(2)
    with left:
        st.subheader("Agent success / failure")
        agent_stats = (
            df.groupby("handler")["success"]
            .agg(total="count", success="sum")
            .assign(failure=lambda x: x["total"] - x["success"])
            .assign(success_rate=lambda x: (100 * x["success"] / x["total"]).round(1))
        )
        st.dataframe(agent_stats, width="stretch")
    with right:
        st.subheader("Messages by category")
        category_counts = df["category"].map(CATEGORY_LABELS).fillna("Unknown").value_counts()
        st.bar_chart(category_counts)

    st.subheader("Ticket actions")
    actions = df[df["ticket_action"].notna()][["created_at", "ticket_id", "ticket_action", "customer_name", "message"]]
    st.dataframe(actions, width="stretch", hide_index=True)

    st.subheader("Interaction history")
    f1, f2 = st.columns(2)
    sources = f1.multiselect("Source", sorted(df["source"].unique()), default=sorted(df["source"].unique()))
    categories = f2.multiselect("Category", list(CATEGORY_LABELS.values()), default=list(CATEGORY_LABELS.values()))
    view = df[df["source"].isin(sources) & df["category"].map(CATEGORY_LABELS).isin(categories)]
    st.dataframe(
        view[["id", "created_at", "source", "customer_name", "message", "category", "classifier_method",
              "agent_path", "response", "ticket_id", "success", "latency_ms", "user_rating"]],
        width="stretch",
        hide_index=True,
    )

    st.subheader("Prompt trace viewer")
    log_id = st.selectbox("Pick an interaction", df["id"].tolist(),
                          format_func=lambda i: f"#{i} · {df.loc[df['id'] == i, 'message'].iloc[0][:70]}")
    row = df[df["id"] == log_id].iloc[0]
    st.json(json.loads(row["prompt_trace"] or "[]"))

    st.subheader("Customer feedback loop")
    rated = df[df["user_rating"].notna()]
    if rated.empty:
        st.caption("No ratings yet. Use 👍 / 👎 under replies in Support Chat.")
    else:
        r1, r2 = st.columns(2)
        r1.metric("👍 Helpful", int((rated["user_rating"] == 1).sum()))
        r2.metric("👎 Not helpful", int((rated["user_rating"] == 0).sum()))
        flagged = rated[rated["user_rating"] == 0][["id", "message", "category", "response"]]
        if not flagged.empty:
            st.markdown("**Replies to review and improve:**")
            st.dataframe(flagged, width="stretch", hide_index=True)

    d1, d2 = st.columns([1, 4])
    d1.download_button("Download logs (CSV)", df.to_csv(index=False), "interaction_logs.csv", "text/csv")
    if d2.button("Clear all logs"):
        db.clear_logs()
        st.rerun()


def page_evaluation():
    st.header("📊 Evaluation")
    st.caption(
        f"Runs {len(TEST_CASES)} labelled test messages through the agents and checks "
        "classification, routing, reply format and reply quality."
    )
    use_judge = st.checkbox(
        "Use LLM-as-judge for empathy and clarity scores",
        value=llm.llm_available(),
        disabled=not llm.llm_available(),
    )
    with st.expander("See test cases"):
        st.dataframe(pd.DataFrame(TEST_CASES), width="stretch", hide_index=True)

    if st.button("Run evaluation", type="primary"):
        bar = st.progress(0.0, "Starting...")
        summary, rows = run_evaluation(use_judge=use_judge, progress=lambda p, m: bar.progress(p, m[:60]))
        bar.empty()
        st.session_state.last_eval = (summary, rows)

    if "last_eval" in st.session_state:
        summary, rows = st.session_state.last_eval
        c = st.columns(5)
        c[0].metric("Classification accuracy", f"{summary['classification_accuracy']}%")
        c[1].metric("Routing success", f"{summary['routing_success_rate']}%")
        c[2].metric("Reply checks passed", f"{summary['response_pass_rate']}%")
        c[3].metric("Empathy (1-5)", summary["avg_empathy"])
        c[4].metric("Clarity (1-5)", summary["avg_clarity"])

        per_cat = pd.Series(summary["per_category_accuracy"]).rename(index=CATEGORY_LABELS)
        st.markdown("**Accuracy by category (%)**")
        st.bar_chart(per_cat)

        results = pd.DataFrame(rows)
        results["expected"] = results["expected"].map(CATEGORY_LABELS)
        results["predicted"] = results["predicted"].map(CATEGORY_LABELS)
        st.dataframe(results, width="stretch", hide_index=True)

    runs = db.list_evaluation_runs()
    if runs:
        st.subheader("Past runs")
        history = pd.DataFrame(runs).drop(columns=["details"])
        st.dataframe(history, width="stretch", hide_index=True)
        if len(history) > 1:
            st.line_chart(
                history.sort_values("id").set_index("id")[
                    ["classification_accuracy", "routing_success_rate", "response_pass_rate"]
                ]
            )


PAGES = {
    "Support Chat": page_chat,
    "Test Scenarios": page_scenarios,
    "Tickets": page_tickets,
    "Logs & Debugging": page_logs,
    "Evaluation": page_evaluation,
}

PAGES[render_sidebar()]()
