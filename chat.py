"""
GrowMate Hydroponics Chatbot — Flask app (like app.py).
Uses hybrid QA: retrieval from hydro_qa_data.json first, then T5/GrowMate fallback.
Run: python chat.py  (serves on http://0.0.0.0:5001, Swagger at /swagger/)
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Paths (match notebook)
BASE_DIR = Path(__file__).resolve().parent
HYDRO_QA_PATH = BASE_DIR / "hydro_qa_data.json"
BATAVIA_DIR = BASE_DIR / "HydroGrowNet of Batavia Dataset" / "all_months_sensory_data"  # optional, for Excel; notebook uses hardcoded list

def clean_text(text: str) -> str:
    """Same as notebook: normalize whitespace for consistent TF-IDF."""
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\r\n]+", " ", text)
    return text

app = Flask(__name__)

# Register "/" before Api() so Flask-RESTX does not override it
CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GrowMate Hydroponics Chatbot</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; max-width: 640px; margin: 2rem auto; padding: 0 1rem; }
    h1 { font-size: 1.5rem; color: #1a5f2a; }
    .input-row { display: flex; gap: 0.5rem; margin: 1rem 0; }
    input[type="text"] { flex: 1; padding: 0.6rem; font-size: 1rem; border: 1px solid #ccc; border-radius: 6px; }
    button { padding: 0.6rem 1.2rem; font-size: 1rem; background: #1a5f2a; color: white; border: none; border-radius: 6px; cursor: pointer; }
    button:hover { background: #2a7f3a; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .response { margin-top: 1rem; padding: 1rem; background: #f5f5f5; border-radius: 8px; white-space: pre-wrap; }
    .mode { font-size: 0.85rem; color: #666; margin-bottom: 0.5rem; }
    .mode.retrieval { color: #1a5f2a; }
    .error { color: #c00; }
    .loading { color: #666; }
  </style>
</head>
<body>
  <h1>GrowMate Hydroponics Chatbot</h1>
  <p>Ask a question about hydroponics. Answers come from curated Q&A (retrieval) or the GrowMate model (generative).</p>
  <div class="input-row">
    <input type="text" id="question" placeholder="e.g. What is the ideal pH for hydroponic lettuce?" />
    <button id="ask">Ask</button>
  </div>
  <div id="out"></div>
  <p style="margin-top: 2rem; font-size: 0.9rem; color: #666;">
    API docs: <a href="/swagger/">/swagger/</a> &nbsp; Health: <a href="/health">/health</a>
  </p>
  <script>
    const question = document.getElementById('question');
    const askBtn = document.getElementById('ask');
    const out = document.getElementById('out');
    function show(msg, cls) {
      out.innerHTML = '<div class="' + (cls || '') + '">' + msg + '</div>';
    }
    askBtn.addEventListener('click', async () => {
      const q = question.value.trim();
      if (!q) { show('Enter a question.', 'error'); return; }
      askBtn.disabled = true;
      show('Thinking...', 'loading');
      try {
        const res = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: q })
        });
        let data;
        try {
          data = await res.json();
        } catch (_) {
          show('Invalid response from server (not JSON).', 'error');
          return;
        }
        if (!res.ok) {
          show('Error: ' + (data.answer || data.message || res.status), 'error');
          return;
        }
        const answer = (data && data.answer != null) ? String(data.answer) : 'No answer returned.';
        show(answer, 'response');
      } catch (e) {
        show('Request failed: ' + e.message, 'error');
      }
      askBtn.disabled = false;
    });
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return CHAT_HTML

api = Api(
    app,
    version="1.0",
    title="GrowMate Hydroponics Chatbot API",
    description="Ask hydroponics questions. Uses retrieval from curated Q&A + T5/GrowMate fallback.",
    doc="/swagger/",
)

# ---------------------------------------------------------------------------
# Load retrieval data: same as notebook (clean_text + same Batavia list)
# ---------------------------------------------------------------------------
def load_hydro_qa(path: Path):
    """Load hydro_qa_data.json - same as notebook (clean_text, min_len=10)."""
    instructions, responses = [], []
    min_len = 10
    if not path.exists():
        return instructions, responses
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        inst = clean_text(item.get("instruction", ""))
        resp = clean_text(item.get("response", ""))
        if len(inst) > min_len and len(resp) > min_len:
            instructions.append(inst)
            responses.append(resp)
    return instructions, responses

# Same hardcoded Batavia Q&A as in the notebook (so retrieval matches notebook)
BATAVIA_QA_HARDCODED = [
    ("What pH range is used in hydroponic lettuce cultivation?",
     "pH typically ranges from 6.5 to 7.0 for hydroponic lettuce in controlled systems."),
    ("What EC levels are used for hydroponic lettuce?",
     "EC (electrical conductivity) for lettuce is typically around 1.4-1.6 mS/cm."),
    ("What water temperature is ideal for hydroponic lettuce?",
     "Water temperature around 22-23°C is common in hydroponic lettuce systems."),
    ("What air temperature and humidity are used for hydroponic lettuce?",
     "Air temperature around 22-24°C and relative humidity 49-61% are typical."),
    ("What CO2 levels are used in hydroponic growth environments?",
     "CO2 levels around 400-450 ppm are typical in indoor hydroponic systems."),
    ("What plant measurements are tracked in hydroponic experiments?",
     "Plant height, shoot length, root length, head diameter, stem diameter, weight, and leaf count."),
    ("What is the typical head diameter for hydroponic lettuce seedlings?",
     "Head diameter in seedling stage is often 3.5-4 cm in controlled hydroponic setups."),
    ("What nutrients and water consumption are monitored in hydroponics?",
     "Nutrient solution, water consumption, pH, EC, TDS, and water temperature are monitored."),
    ("What environmental parameters does the Batavia hydroponic dataset include?",
     "pH, EC, TDS, water temp, air temp, RH, CO2, plant height, shoot/root length, weight."),
]

hydro_instructions, hydro_responses = load_hydro_qa(HYDRO_QA_PATH)
print(f"Loaded hydro_qa: {len(hydro_instructions)} pairs (same as notebook)")

batavia_instructions = [q for q, a in BATAVIA_QA_HARDCODED]
batavia_responses = [a for q, a in BATAVIA_QA_HARDCODED]
print(f"Batavia Q&A: {len(batavia_instructions)} pairs (same as notebook)")

# Build TF-IDF index over hydro_qa + Batavia questions
retrieval_questions = hydro_instructions + batavia_instructions
retrieval_answers = hydro_responses + batavia_responses
tfidf_vectorizer = None
tfidf_matrix = None

if len(retrieval_questions) == 0:
    print("No retrieval data available. Make sure hydro_qa and Batavia data are loaded.")
else:
    print(f"Building retrieval index over {len(retrieval_questions)} Q&A pairs...")
    tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(retrieval_questions)
    print("Retrieval index ready.")

# ---------------------------------------------------------------------------
# Load T5 model (GrowMate) for generative fallback
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
tokenizer = None
MODEL_NAME = "Afsa20/Farmsmart_Growmate"

def _load_model():
    global model, tokenizer
    if model is not None:
        return
    from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
    print(f"Loading {MODEL_NAME}...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    config = T5Config.from_pretrained(MODEL_NAME)
    config.tie_word_embeddings = False
    model = T5ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        config=config,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type == "cpu":
        model = model.to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Model loaded.")

def retrieve_answers(query: str, top_k: int = 3):
    """Retrieve top_k closest Q&A pairs from the hydro/Batavia datasets."""
    if len(retrieval_questions) == 0 or tfidf_vectorizer is None or tfidf_matrix is None:
        return []
    query_vec = tfidf_vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [
        {"score": float(sims[i]), "question": retrieval_questions[i], "answer": retrieval_answers[i]}
        for i in top_idx
    ]

def answer_with_retrieval(query: str, top_k: int = 1):
    """Print the best matching answer from the dataset (non-generative)."""
    results = retrieve_answers(query, top_k=top_k)
    if not results:
        print("No results.")
        return
    best = results[0]
    print(f"Best match score: {best['score']:.3f}\n")
    print(best["question"])
    print(best["answer"])

def generate_response(question: str, max_length: int = 150) -> str:
    try:
        _load_model()
        inp = f"Answer this hydroponic farming question: {question}"
        inputs = tokenizer(inp, return_tensors="pt", max_length=512, truncation=True).to(device)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                min_length=15,
                num_beams=4,
                early_stopping=True,
                repetition_penalty=2.5,
                no_repeat_ngram_size=3,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return (text or "").strip() or "No answer generated."
    except Exception as e:
        print(f"Generate error: {e}")
        return f"Could not generate an answer (model error: {e!s}). Try rephrasing or check the retrieval result above."

def answer_hybrid(question: str, thresh: float = 0.4, top_k_retrieval: int = 3, use_print: bool = False):
    """Retrieval first; if score >= thresh use dataset answer, else GrowMate. Returns (answer, mode, score_or_None)."""
    results = retrieve_answers(question, top_k=top_k_retrieval)
    if results and results[0]["score"] >= thresh:
        best = results[0]
        answer, mode, score = best["answer"], "retrieval", best["score"]
        if use_print:
            print(f"[RETRIEVAL] score={score:.3f}\n")
            print(best["question"])
            print(answer)
        ans = (answer or "").strip()
        return ans or "No answer in dataset.", mode, score
    answer = generate_response(question)
    if use_print:
        print("[GENERATIVE] Using GrowMate model (fallback).\n")
        print(answer)
    ans = (answer or "").strip()
    return ans or "No answer generated.", "generative", None

# ---------------------------------------------------------------------------
# API (Swagger)
# ---------------------------------------------------------------------------
chat_request = api.model("ChatRequest", {
    "question": fields.String(required=True, description="Hydroponics question"),
})
chat_response = api.model("ChatResponse", {
    "answer": fields.String(description="Answer text"),
    "mode": fields.String(description="retrieval or generative"),
    "score": fields.Float(description="Retrieval similarity score, or null if generative"),
})

@api.route("/chat")
class Chat(Resource):
    @api.expect(chat_request)
    def post(self):
        """Ask the GrowMate hydroponics chatbot a question."""
        try:
            data = request.get_json(force=True, silent=True) or {}
            question = (data.get("question") or "").strip()
            if not question:
                return jsonify({"answer": "Please provide a question.", "mode": "error", "score": None}), 400
            answer, mode, score = answer_hybrid(question)
            answer = (answer or "").strip() or "No answer available."
            return jsonify({"answer": answer, "mode": mode, "score": score})
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Chat error: {e}")
            return jsonify({"answer": f"Server error: {e!s}", "mode": "error", "score": None}), 500

@api.route("/health")
class Health(Resource):
    def get(self):
        """Health check."""
        return {
            "status": "healthy",
            "retrieval_pairs": len(retrieval_questions),
            "model_loaded": model is not None,
        }

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
