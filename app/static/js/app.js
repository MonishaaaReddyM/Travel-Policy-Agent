const employeeIdInput = document.getElementById("employee-id");
const questionInput = document.getElementById("question");
const askBtn = document.getElementById("ask-btn");
const clearBtn = document.getElementById("clear-btn");
const loadingEl = document.getElementById("loading");
const errorEl = document.getElementById("error");
const answerSection = document.getElementById("answer-section");
const answerText = document.getElementById("answer-text");
const sourceBlock = document.getElementById("source-block");
const sourceText = document.getElementById("source-text");
const routeBadge = document.getElementById("route-badge");
const historyEl = document.getElementById("history");

function showError(message) {
    errorEl.textContent = message;
    errorEl.classList.remove("hidden");
}

function clearError() {
    errorEl.classList.add("hidden");
    errorEl.textContent = "";
}

function setLoading(isLoading) {
    loadingEl.classList.toggle("hidden", !isLoading);
    askBtn.disabled = isLoading;
}

function renderAnswer(data) {
    answerSection.classList.remove("hidden");
    answerText.textContent = data.answer;

    if (data.sources && data.sources.length > 0) {
        sourceBlock.classList.remove("hidden");
        sourceText.textContent = data.sources.join(", ");
    } else {
        sourceBlock.classList.add("hidden");
    }

    routeBadge.textContent = data.route ? data.route.replace(/_/g, " ") : "";
}

function renderHistory(history) {
    if (!history || history.length === 0) {
        historyEl.innerHTML = '<p class="empty-state">No conversation yet — ask a question above.</p>';
        return;
    }

    historyEl.innerHTML = "";
    for (const turn of history) {
        const turnEl = document.createElement("div");
        turnEl.className = "history-turn";

        const q = document.createElement("div");
        q.className = "history-question";
        q.textContent = "User: " + turn.question;

        const a = document.createElement("div");
        a.className = "history-answer";
        a.textContent = "Assistant: " + turn.answer;

        const t = document.createElement("div");
        t.className = "history-time";
        t.textContent = turn.timestamp || "";

        turnEl.appendChild(q);
        turnEl.appendChild(a);
        turnEl.appendChild(t);
        historyEl.appendChild(turnEl);
    }

    // keep the latest turn in view
    historyEl.scrollTop = historyEl.scrollHeight;
}

async function askAssistant() {
    const employeeId = employeeIdInput.value.trim();
    const question = questionInput.value.trim();

    if (!question) {
        showError("Please enter a question before asking.");
        return;
    }

    clearError();
    setLoading(true);
    answerSection.classList.add("hidden");

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ employee_id: employeeId, question: question }),
        });

        const data = await response.json();

        if (!response.ok) {
            showError(data.error || "Something went wrong. Please try again.");
            return;
        }

        renderAnswer(data);
        renderHistory(data.history);
        questionInput.value = "";
    } catch (err) {
        showError("Could not reach the assistant. Please check your connection and try again.");
    } finally {
        setLoading(false);
    }
}

async function clearConversation() {
    try {
        await fetch("/clear", { method: "POST" });
    } catch (err) {
        // Non-critical - clear the UI regardless
    }
    answerSection.classList.add("hidden");
    clearError();
    renderHistory([]);
}

askBtn.addEventListener("click", askAssistant);
clearBtn.addEventListener("click", clearConversation);
questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        askAssistant();
    }
});