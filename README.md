# GrowMate Hydroponics Assistant & EzaSmart API

End‑to‑end hydroponics assistant combining:

- **GrowMate Hydroponics Chatbot** – hybrid QA system for hydroponics questions (retrieval + `Afsa20/Farmsmart_Growmate` T5 model).
- **EzaSmart Hydroponics API** – Random Forest–based recommendation service for sensor data (crop + pH + EC + temperature → recommended action).

This repo contains the **ML notebook**, **APIs**, and **scrapers** used to build the minimum viable product for this capstone project.

**Demo video:** [Initial Software Demo](https://www.youtube.com/watch?v=_n-pDjz9OK0) (YouTube)

---

## 1. Repository structure

- `chatbot_training_notebook.ipynb` – main ML notebook (data engineering, visualization, model architecture, metrics, hybrid QA logic).
- `scrape.py` – StackExchange (Gardening) scraper for hydroponics Q&A → `hydro_qa_data.json`.
- `hydro_qa_data.json` – curated hydroponics Q&A pairs (instruction / response / source) from Gardening.StackExchange.
- `HydroGrowNet of Batavia Dataset/` – Excel files with environmental + plant measurements (used to hand‑craft Batavia Q&A in the notebook).
- `app.py` – **EzaSmart** Flask + Flask‑RESTX API (Random Forest model for sensor‑based actions, Swagger at `/swagger/`).
- `chat.py` – **GrowMate chatbot** Flask + Flask‑RESTX API + simple web UI (hybrid QA, Swagger at `/swagger/` and chat UI at `/`).
- `Results/` – trained artefacts for EzaSmart: `random_forest_model.pkl`, `feature_scaler.pkl`, `crop_encoder.pkl`, `action_encoder.pkl` (loaded by `app.py`). 

---

## 2. Datasets used

### 2.1 GrowMate chatbot notebook

The chatbot notebook brings together two main data sources:

1. **`hydro_qa_data.json` (StackExchange Q&A)**  
   - Built by `scrape.py` using the StackAPI client on the **Gardening.StackExchange** site.  
   - Filters questions tagged with `hydroponic` and fetches the **accepted answer** – or the top‑voted answer if no accepted answer exists.  
   - For each item it stores:
     - `instruction`: cleaned concatenation of the question **title + body**.
     - `response`: cleaned answer body (HTML → text via BeautifulSoup, whitespace normalized).
     - `source`: original question URL (StackExchange link).
   - Used as the **primary retrieval corpus** for the chatbot.

2. **HydroGrowNet of Batavia Dataset (Excel files)**  
   - Directory: `HydroGrowNet of Batavia Dataset/all_months_sensory_data/*.xlsx`.  
   - Contains **environmental conditions** (pH, EC, TDS, water/air temperature, RH, CO₂) and **plant measurements** (height, weight, leaf counts, etc.) for hydroponic lettuce experiments.  
   - In the notebook, you do **lightweight data engineering** to craft a small set of **canonical Q&A pairs** such as:
     - “What pH range is used in hydroponic lettuce cultivation?”  
     - “What environmental parameters does the Batavia hydroponic dataset include?”  
   - These are appended to the StackExchange corpus so that the chatbot can answer dataset‑specific questions.

In the notebook, these sources become:

- `hydro_instructions`, `hydro_responses` – from `hydro_qa_data.json`.  
- `batavia_instructions`, `batavia_responses` – from handcrafted Batavia Q&A.  

They are combined into `all_instructions` / `all_targets` for dataset creation and also into `retrieval_questions` / `retrieval_answers` for the TF‑IDF retrieval index.

### 2.2 EzaSmart Hydroponics API

The `app.py` service exposes a Random Forest classifier trained (offline, in a separate notebook) on a **hydroponic sensor dataset** with schema roughly:

- **Features**:
  - `crop_id` – categorical (e.g. `Lettuce`, `Peppers`, `Tomatoes`).  
  - `ph_level` – numeric (4.0–8.5).  
  - `ec_value` – numeric EC (mS/cm, 0.5–4.0).  
  - `ambient_temp` – numeric temperature (°C, 15–32).
- **Label** (action to take):
  - e.g. `Add_pH_Down`, `Add_pH_Up`, `Add_Nutrients`, `Dilute`, `Maintain`.

During training (notebook not included here to keep the repo focused), you:

- Encode `crop_id` and `action` via `crop_encoder.pkl` / `action_encoder.pkl`.  
- Standardize numeric features via `feature_scaler.pkl`.  
- Fit a `RandomForestClassifier` and persist it as `random_forest_model.pkl`.  

At runtime, `app.py` loads these artefacts and exposes `/predict` as a REST endpoint with full validation and a human‑readable recommendation string.

---

## 3. Environment setup

### 3.1 Python & dependencies

Recommended: Python **3.10–3.12** with a virtual environment.

Install core dependencies (you can adapt this list to a `requirements.txt` if you prefer):

```bash
pip install \
  torch transformers datasets accelerate evaluate \
  scikit-learn pandas numpy matplotlib tqdm \
  stackapi beautifulsoup4 openpyxl \
  flask flask-restx
```

> Note: Hugging Face models (GrowMate) are large; ensure you have a stable internet connection and sufficient RAM.

### 3.2 Data files

Place the following in the project root:

- `hydro_qa_data.json` – built via `scrape.py` or provided.  
- `HydroGrowNet of Batavia Dataset/` – Excel directory (optional if only StackExchange data is needed).  
- For EzaSmart: place `random_forest_model.pkl`, `feature_scaler.pkl`, `crop_encoder.pkl`, `action_encoder.pkl` in the `Results/` directory (loaded by `app.py` from there).

---

## 4. How to run

### 4.1 GrowMate chatbot notebook

1. Open `chatbot_training_notebook.ipynb` in Jupyter / VS Code.
2. Run cells in order:
   - **§1–4**: Model load + data engineering (StackExchange + Batavia Q&A).  
   - **§5**: Train/Val/Test split & tokenization (builds `train_ds`, `val_ds`, `test_ds` + `data_collator`).  
   - **§6**: Quick test with `generate_response` (base GrowMate).  
   - **§6 (Retrieval)**: Build TF‑IDF index and define `answer_with_retrieval` and `answer_hybrid`.  
   - **§7**: Data Exploration & Visualization (run for plots).  
   - **§8**: Read Model Architecture description.  
   - **§9**: Run small 1‑epoch fine‑tune + ROUGE evaluation for initial metrics.
3. Use `answer_hybrid("<question>")` to demo the hybrid QA behavior.

### 4.2 GrowMate chatbot API & web UI (`chat.py`)

```bash
python chat.py
```

Then open:

- **Chat UI**: `http://localhost:5001/` – ask questions, see whether answers came from **retrieval** or **GrowMate**.  
- **Swagger docs**: `http://localhost:5001/swagger/` – interactive docs for the `/chat` and `/health` endpoints.

### 4.3 EzaSmart Hydroponics API (`app.py`)

```bash
python app.py
```

Then open:

- **Swagger docs**: `http://localhost:5000/swagger/` – try the `/predict` endpoint by sending sensor readings.  
- **Health check**: `http://localhost:5000/health`.

---