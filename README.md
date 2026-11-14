# Banking Support AI Agents

This is my capstone project for the Applied Generative AI Specialization. It's a small customer support desk for a pretend bank, run by three AI "agents" that work together. A customer types a message, and the agents figure out what kind of message it is and reply the right way.

It's built with Python, Streamlit and SQLite, and uses OpenAI for the language model parts.

> This is an academic project. Please don't type real bank or personal details into it.

## What it does

When a message comes in, it goes through a short chain of agents:

1. **Classifier Agent** reads the message and decides if it's
   - positive feedback ("Thanks for fixing my login!"),
   - negative feedback ("My card still hasn't arrived."), or
   - a query ("What's the status of ticket 650932?").
2. Then the message is passed to the right helper:
   - **Feedback Handler (positive)** writes a warm thank-you reply.
   - **Feedback Handler (negative)** creates a new 6-digit support ticket in the database and says sorry.
   - **Query Handler** pulls the ticket number out of the message, looks it up, and tells the customer its status.

A few examples:

| Customer says | Goes to | Reply |
|---|---|---|
| Thanks for sorting out my net banking login issue. | Positive feedback | Thank you for your kind words! We're delighted to assist you. |
| My debit card replacement still hasn't arrived. | Negative feedback | We apologize for the inconvenience. A new ticket #784521 has been generated, and our team will follow up shortly. |
| Could you check the status of ticket 650932? | Query | Your ticket #650932 is currently marked as: Resolved. |

## The app

The Streamlit app has five pages, which you can pick from the sidebar:

- **Support Chat** – talk to the agents like a customer. Open "Agent details" under any reply to see how it was classified, which agents handled it, what happened in the database, and the full prompt trace. You can rate replies with 👍 / 👎.
- **Test Scenarios** – ready-made messages for each agent. Click "Run" on one, or "Run all scenarios".
- **Tickets** – everything in the `support_tickets` table. You can change a ticket's status here (e.g. Open → Resolved) and then ask about it in the chat.
- **Logs & Debugging** – every interaction is saved. This page shows success/failure rate per agent, ticket actions, response times, the prompt trace for any message, and the 👎 replies that need a second look.
- **Evaluation** – runs 16 labelled test messages through the system and reports classification accuracy, routing success rate, how many replies passed the format/database checks, and empathy and clarity scores (1–5) given by an LLM judge. Past runs are kept so you can compare them.

### No API key? It still works

If there's no OpenAI key, the app switches to a simple keyword-based classifier and template replies. It's less clever, but everything still works, which is handy for trying it out. The sidebar tells you which mode you're in.

## Project files

```
app.py            the Streamlit app (all five pages)
agents.py         the Classifier, Feedback Handler and Query Handler agents, plus the orchestrator
database.py       SQLite setup: tickets, interaction logs and evaluation runs
llm.py            small wrapper around the OpenAI API, reads the key from secrets or .env
evaluation.py     test cases and scoring
requirements.txt  Python packages
.env.example      template for your local .env file
```

The database file is created on its own at `data/support.db` the first time the app runs, with a few sample tickets (650932, 123456, 456789, 112233, 998877) so you can try queries right away.

## Running it on your computer

You need Python 3.10 or newer.

```bash
git clone https://github.com/bvenkatsr/banking-support-ai-agents.git
cd banking-support-ai-agents
python -m venv .venv
.venv\Scripts\activate          # on Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and put your OpenAI key in it:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

Then start the app:

```bash
streamlit run app.py
```

It opens at http://localhost:8501.

## Putting it on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **Create app**, pick this repo, branch `main`, and main file `app.py`.
3. Deploy it.
4. Once it's up, open the app's **Settings → Secrets** and paste:

   ```toml
   OPENAI_API_KEY = "sk-..."
   OPENAI_MODEL = "gpt-4o-mini"
   ```

5. Save. The app restarts and the sidebar should say "OpenAI connected".

One thing to know: Streamlit Cloud doesn't keep files forever, so the SQLite database resets whenever the app restarts or goes to sleep. That's fine for a demo, since the sample tickets come back automatically.

## Keeping the key safe

`.env`, `.streamlit/secrets.toml` and the database file are all in `.gitignore`, so your key never ends up on GitHub. Only the `.example` files are committed.
