"""
app/app.py
----------
Day 4 deliverable, Part 3: Flask web application - the presentation layer
for the AI backend built on Days 1-3.

    Browser -> POST /ask -> Flask -> Agent -> RAG / Tools / Memory -> LLM
            -> Response -> Flask JSON response -> Browser

This file intentionally stays thin: routes parse the request, call into
src/ (agent.py, memory.py), and shape the response. All the actual AI
logic (retrieval, tool calls, routing decisions) lives in src/, not here -
see the "Recommended Separation of Responsibilities" section of the
project brief.

Run directly:
    python app/app.py
Then open http://localhost:5000 in a browser.
"""

import sys
import uuid
from pathlib import Path

from flask import Flask, render_template, request, jsonify, session

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import run_agent, EMPLOYEE_ID_PATTERN  # noqa: E402
from memory import ConversationMemory, resolve_followup_question  # noqa: E402

app = Flask(__name__)
app.secret_key = "dev-secret-key-change-if-deploying"  # fine for a local training project

# In-memory session store: {session_id: ConversationMemory}. Simple and
# sufficient for a single-process training app; a production deployment
# would back this with Redis or a database instead, and use a real
# app.secret_key from an environment variable rather than a hardcoded one.
_SESSIONS: dict[str, ConversationMemory] = {}


def _get_memory() -> ConversationMemory:
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    session_id = session["session_id"]
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = ConversationMemory()
    return _SESSIONS[session_id]


def _combine_employee_id(question: str, employee_id: str) -> str:
    """
    The UI has a separate Employee ID field from the free-text question box
    (e.g. question = "Can I take an airport trip costing ₹2,500?" with no
    "EMP001" in the text itself). If the employee ID isn't already
    mentioned in the question, fold it in so the agent's entity extraction
    picks it up.
    """
    employee_id = (employee_id or "").strip().upper()
    if not employee_id:
        return question
    if EMPLOYEE_ID_PATTERN.search(question):
        return question  # already mentioned in the question text
    return f"For employee {employee_id}, {question}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    employee_id = (data.get("employee_id") or "").strip()
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Please enter a question before asking."}), 400

    memory = _get_memory()

    try:
        combined_question = _combine_employee_id(question, employee_id)
        resolved_question = resolve_followup_question(combined_question, memory)
        result = run_agent(resolved_question)
    except FileNotFoundError as e:
        # e.g. the vector store hasn't been built yet (embeddings.py not run)
        app.logger.error(f"Missing data/index: {e}")
        return jsonify({
            "error": "The assistant's knowledge base isn't ready yet. "
                     "Please contact the administrator."
        }), 503
    except Exception as e:
        # Catch-all: never leak a raw traceback to the user.
        app.logger.exception("Unhandled error in /ask")
        return jsonify({
            "error": "Something went wrong while answering that question. Please try again."
        }), 500

    # Store the ORIGINAL question (not the internally-resolved one) so the
    # conversation history shown to the user reads naturally.
    memory.add_turn(question, result.final_answer)

    return jsonify({
        "answer": result.final_answer,
        "route": result.route,
        "sources": result.sources,
        "history": memory.as_list(),
    })


@app.route("/clear", methods=["POST"])
def clear():
    memory = _get_memory()
    memory.clear()
    return jsonify({"status": "cleared"})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)