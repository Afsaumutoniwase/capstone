# GrowMate & EzaSmart Hydroponics Assistant

**Bridging Knowledge Gaps with Hydroponics Software System**  
Capstone Project - ML Track

**Author:** Afsa Umutoniwase  
**Date:** February 2026  
**Demo Video:** [Watch Demo](https://drive.google.com/file/d/10u1vyFhdz7RXZDg26ieQW9BG9m67Rxjd/view?usp=drive_link)  
**Repository:** [github.com/Afsaumutoniwase/capstone](https://github.com/Afsaumutoniwase/capstone)

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Model Performance](#model-performance)
- [Environment Setup](#environment-setup)
- [Running the Notebooks](#running-the-notebooks)
- [Running the APIs](#running-the-apis)
- [API Documentation](#api-documentation)

## Overview

This project combines two machine learning systems for hydroponic farming:

### GrowMate Chatbot
A hybrid question-answering system that combines retrieval-based search with generative AI to answer hydroponics questions.

- **Base Model:** Afsa20/Farmsmart_Growmate (T5 transformer, 60M parameters)
- **Approach:** TF-IDF retrieval with T5 generation fallback
- **Knowledge Base:** 124 Q&A pairs (115 from StackExchange + 9 from Batavia dataset)
- **Performance:** ROUGE-1: 0.64, ROUGE-2: 0.48, ROUGE-L: 0.60
- **Notebook:** `chatbot_training_notebook.ipynb`
- **API:** `chat.py` (Flask REST API with web UI)

### EzaSmart Sensor Monitor
A classification system that recommends actions based on hydroponic sensor readings.

- **Model:** Random Forest Classifier (100 trees)
- **Input:** Crop type, pH level, EC value, ambient temperature
- **Output:** One of 5 actions (Add_pH_Down, Add_pH_Up, Add_Nutrients, Dilute, Maintain)
- **Performance:** 97.8% accuracy, 0.97 weighted F1-score
- **Training Data:** 5,000+ samples from IoT sensors and synthetic data
- **Notebook:** `EzaSmart_ML_Model_Notebook.ipynb`
- **API:** `app.py` (Flask REST API)

## Repository Structure

```
capstone/
├── chatbot_training_notebook.ipynb    # GrowMate training, visualization, evaluation
├── EzaSmart_ML_Model_Notebook.ipynb   # EzaSmart data engineering, model training
├── app.py                              # EzaSmart Flask REST API (port 5000)
├── chat.py                             # GrowMate Flask API + web UI (port 5001)
├── scrape.py                           # StackExchange data scraper
├── hydro_qa_data.json                  # 115 Q&A pairs from StackExchange
├── requirements.txt                    # Python dependencies
├── HydroGrowNet of Batavia Dataset/    # Environmental sensor data (Excel files)
├── Kaggle data/                        # IoTData --Raw--.csv
├── Results/                            # Trained EzaSmart model artifacts
│   ├── random_forest_model.pkl
│   ├── feature_scaler.pkl
│   ├── crop_encoder.pkl
│   └── action_encoder.pkl
└── trained_chatbot_model/              # Fine-tuned GrowMate model + visualizations
    ├── pytorch_model.bin
    ├── config.json
    ├── tokenizer files
    └── 5 visualization PNGs
```

## Datasets

### GrowMate Chatbot Datasets

**1. hydro_qa_data.json (StackExchange)**
- **Source:** Gardening StackExchange, scraped using `scrape.py`
- **Content:** 115 Q&A pairs tagged with "hydroponics"
- **Fields:** 
  - `instruction`: Question (title + body, cleaned)
  - `response`: Answer (accepted or top-voted, HTML converted to text)
  - `source`: Original StackExchange URL
- **Use:** Primary retrieval corpus for TF-IDF search

**2. HydroGrowNet of Batavia Dataset**
- **Location:** `HydroGrowNet of Batavia Dataset/all_months_sensory_data/*.xlsx`
- **Content:** Environmental conditions and plant measurements for hydroponic lettuce
- **Parameters:** pH, EC, TDS, water/air temperature, humidity, CO2, plant height, weight, leaf count
- **Use:** Generated 9 canonical Q&A pairs about dataset-specific parameters
- **Examples:**
  - "What pH range is used in hydroponic lettuce cultivation?"
  - "What environmental parameters does the Batavia dataset include?"

**Combined:** 124 Q&A pairs used for retrieval index and training/validation/test splits (78% / 11% / 11%)

### EzaSmart Sensor Monitor Datasets

**1. IoT Sensor Data (Kaggle)**
- **Source:** `Kaggle data/IoTData --Raw--.csv`
- **Content:** Real hydroponics IoT sensor and actuator logs
- **Raw Features:** pH, TDS, DHT_temp, pH_reducer, nutrients_adder, add_water
- **Processing:**
  - Converted TDS to EC (TDS / 500)
  - Created target labels based on pH/EC thresholds and actuator states
  - Filtered unrealistic values (pH 3-9, EC 0.1-5.0, temp 10-40°C)

**2. Synthetic Data (Rwanda Context)**
- **Records:** 5,000 samples
- **Generation Method:**
  - Based on Oklahoma State University Extension hydroponic standards
  - Calibrated for Rwanda Southern Province climate (15-32°C)
  - Added measurement noise (±0.2) to simulate handheld meter accuracy
- **Crops:** Lettuce, Peppers, Tomatoes with optimal ranges
- **Scenario Distribution:** 40% optimal, 60% problematic conditions
- **Use:** Address limited real-world data and balance action classes

**Combined:** Final dataset merges IoT and synthetic data with features: Crop_ID, pH_Level, EC_Value, Ambient_Temp

## Model Performance

### GrowMate Chatbot

**ROUGE Metrics (Validation Set):**
- ROUGE-1: 0.64 (unigram overlap)
- ROUGE-2: 0.48 (bigram overlap)
- ROUGE-L: 0.60 (longest common subsequence)

**Model Details:**
- Parameters: 60 million (T5-based)
- Training: 1 epoch (demonstration), learning rate 5e-5
- Hybrid System: Retrieval (similarity threshold 0.4) + Generation fallback

### EzaSmart Sensor Monitor

**Classification Metrics (Test Set):**
- Accuracy: 97.8%
- Precision: 0.97 (weighted average)
- Recall: 0.97 (weighted average)
- F1-Score: 0.97 (weighted average)

**Comparison:**
- Random Forest: 97.8% accuracy (selected model)
- Logistic Regression: 89.2% accuracy (baseline)

**Feature Importance:** EC_Value and pH_Level are dominant predictors

## Environment Setup

### Prerequisites

- Python 3.10 or higher
- 8GB RAM minimum (16GB recommended for model training)
- Internet connection (for downloading HuggingFace models)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Afsaumutoniwase/capstone.git
cd capstone
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

This installs:
- PyTorch 2.0+ (deep learning)
- Transformers 4.25+ (HuggingFace T5 models)
- Scikit-learn 1.3.2 (Random Forest, preprocessing)
- Flask 3.0.0 + Flask-RESTX 1.3.0 (REST APIs)
- Pandas, NumPy, Matplotlib, Seaborn (data processing & visualization)
- Additional libraries: datasets, accelerate, evaluate, rouge-score, tqdm, openpyxl

## Running the Notebooks

### 1. GrowMate Chatbot Training Notebook

**File:** `chatbot_training_notebook.ipynb`

**How to Run:**

1. Open in Jupyter Notebook or VS Code
2. **Update the BASE_DIR path** (Cell 4):
   ```python
   BASE_DIR = Path(r"C:\Users\HP\Desktop\ALU\capstone")  # Change to your path
   ```

3. Run cells sequentially:

**Section 1: Load Base Model**
- Loads Afsa20/Farmsmart_Growmate from HuggingFace
- Configures tokenizer and model for T5 architecture

**Section 2-3: Load & Process Data**
- Loads `hydro_qa_data.json` (115 Q&A pairs)
- Generates Batavia dataset Q&A (9 pairs)
- Combines into 124 total pairs

**Section 4: Create Instruction-Formatted Dataset**
- Formats questions with "Answer this hydroponic farming question:" prefix
- Prepares for T5 training

**Section 5: Data Exploration & Visualization**
- Cell 14: Creates 9-subplot comprehensive data dashboard
- Section 5.1: Keyword analysis (top 25 words in questions/answers)
- Section 5.2: Data quality metrics (lengths, vocabulary, question types)
- Saves: `01_data_dashboard.png`, `02_keyword_analysis.png`, `03_data_quality_metrics.png`

**Section 6: Train/Val/Test Split & Tokenization**
- Splits: 78% train, 11% validation, 11% test
- Tokenizes with max input 512, max target 256 tokens

**Section 6 (Training):**
- Runs 1-epoch demonstration training
- Computes ROUGE scores on validation set
- **Saves trained model to** `trained_chatbot_model/`
- Visualization: `04_training_metrics.png`

**Section 7: Test Predictions**
- Tests model on 8 sample questions
- Generates responses and analyzes lengths
- Visualization: `05_test_predictions.png`

**Section 8: Retrieval-Based QA**
- Builds TF-IDF index over all Q&A pairs
- Implements `retrieve_answers()` and `answer_with_retrieval()`

**Section 8 (Hybrid QA):**
- Implements `answer_hybrid()`: retrieval first, generation fallback
- Demo examples show retrieval vs generative responses

**Section 9: Summary**
- Lists all saved outputs (5 PNGs + model files)

**Expected Outputs:**
- 5 visualization PNG files in `trained_chatbot_model/`
- Trained model files: `pytorch_model.bin`, `config.json`, tokenizer files
- Console output showing ROUGE scores and sample predictions

### 2. EzaSmart Sensor Monitor Notebook

**File:** `EzaSmart_ML_Model_Notebook.ipynb`

**How to Run:**

1. Open in Jupyter Notebook or VS Code
2. **Update the BASE_DIR path** (Cell 2):
   ```python
   BASE_DIR = Path(r"C:\Users\HP\Desktop\ALU\capstone")  # Change to your path
   ```

3. Run cells sequentially:

**Section 1: Data Loading**
- Loads IoT sensor data from `Kaggle data/IoTData --Raw--.csv`
- Displays data shape and overview

**Section 2: Data Engineering**
- Cell 6: Processes IoT data
  - Converts pH, TDS, temperature to numeric
  - Converts TDS to EC (TDS / 500)
  - Creates `create_target_action()` function based on pH/EC thresholds
  - Filters unrealistic values
- Cell 7: Generates synthetic data
  - 5,000 samples for Lettuce, Peppers, Tomatoes
  - Scenario-based generation (optimal, high/low pH, high/low EC)
  - Adds measurement noise, clips to realistic ranges
- Cell 8: Merges datasets
  - Assigns crop types to IoT data based on pH/EC ranges
  - Concatenates IoT + synthetic into final dataset

**Section 3: Data Visualization**
- Cell 9-12: Creates comprehensive visualizations
  - pH, EC, temperature distributions
  - Target action distribution (bar chart)
  - Correlation matrix heatmap
  - pH vs EC scatter plot by crop type
  - Crop-specific analysis (4 subplots)
  - Box plots for features by target action
- Saves: `data_distributions.png`, `correlation_matrix.png`, `ph_ec_by_crop.png`, etc.

**Section 4: Model Architecture**
- Cell 13: Prepares features and labels
  - Encodes categorical variables (Crop_ID, Target_Action)
  - Splits 80% train, 20% test
  - Displays feature columns and class counts
- Cell 14: Trains Random Forest (100 trees)
  - Shows feature importance
- Cell 15: Trains Logistic Regression (baseline)
  - Scales features with StandardScaler
- Cell 16: Visualizes feature importance comparison

**Section 5: Performance Metrics**
- Cell 17: Calculates metrics for both models
  - Accuracy, precision, recall, F1-score
- Cell 18: Visualizes performance
  - Metrics comparison bar chart
  - Confusion matrix for Random Forest
- Cell 19: Prints detailed classification reports

**Section 6: Save Models**
- Saves to `Results/` directory:
  - `random_forest_model.pkl`
  - `logistic_regression_model.pkl`
  - `feature_scaler.pkl`
  - `crop_encoder.pkl`
  - `action_encoder.pkl`
  - `model_metadata.json` (performance metrics)

**Expected Outputs:**
- Multiple visualization PNG files in `Results/`
- 5 trained model pickle files
- Console output showing 97.8% accuracy for Random Forest

## Running the APIs

### 1. Start EzaSmart API

```bash
python app.py
```

**Server Details:**
- Port: 5000
- Host: 0.0.0.0 (accessible from network)
- Debug mode: ON

**Endpoints:**
- Swagger UI: http://localhost:5000/swagger/
- Health check: http://localhost:5000/health
- Prediction: POST http://localhost:5000/predict

**On startup, you should see:**
```
Models loaded successfully!
 * Running on http://127.0.0.1:5000
```

### 2. Start GrowMate Chatbot API

```bash
python chat.py
```

**Server Details:**
- Port: 5001
- Host: 0.0.0.0
- Debug mode: ON

**Endpoints:**
- Web UI: http://localhost:5001/
- Swagger UI: http://localhost:5001/swagger/
- Health check: http://localhost:5001/health
- Chat: POST http://localhost:5001/chat

**On startup, you should see:**
```
Loaded hydro_qa: 115 pairs
Batavia Q&A: 9 pairs
Building retrieval index over 124 Q&A pairs...
Retrieval index ready.
 * Running on http://127.0.0.1:5001
```

**Note:** First run will download the T5 model from HuggingFace (may take several minutes)

## API Documentation

### EzaSmart API (port 5000)

**POST /predict**

Accepts sensor readings and returns recommended action.

**Request Body:**
```json
{
  "Crop_ID": "Lettuce",
  "pH_Level": 7.2,
  "EC_Value": 1.8,
  "Ambient_Temp": 24.5
}
```

**Response:**
```json
{
  "prediction": "Add_pH_Down",
  "recommendation": "Action: Add pH Down solution to lower pH",
  "confidence": 0.98,
  "input": {
    "Crop_ID": "Lettuce",
    "pH_Level": 7.2,
    "EC_Value": 1.8,
    "Ambient_Temp": 24.5
  }
}
```

**Valid Crop_ID values:** Lettuce, Peppers, Tomatoes  
**Valid ranges:** pH 3.0-9.0, EC 0.1-5.0, Temperature 10-40°C

### GrowMate Chatbot API (port 5001)

**POST /chat**

Sends question and receives answer with source information.

**Request Body:**
```json
{
  "question": "What is the ideal pH for hydroponic lettuce?"
}
```

**Response (Retrieval Mode):**
```json
{
  "answer": "The optimal pH range for hydroponic lettuce is 5.5 to 6.5...",
  "mode": "retrieval",
  "confidence": 0.87,
  "source": "Retrieved from knowledge base"
}
```

**Response (Generative Mode):**
```json
{
  "answer": "Hydroponic lettuce grows best at pH 5.5-6.5...",
  "mode": "generative",
  "confidence": null,
  "source": "Generated by GrowMate AI"
}
```

**Web UI:** Visit http://localhost:5001/ for interactive chat interface

## Notes

- Both notebooks include comprehensive visualizations documenting all development stages
- All saved images are static PNG files (GitHub-compatible)
- Model files are production-ready and can be deployed in larger systems
- APIs use Flask development server (use production WSGI server for deployment)
- Notebooks are cleaned of widget metadata and render correctly on GitHub
