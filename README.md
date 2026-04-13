# 🚨 AI-Powered Defect Analysis Dashboard

## 📌 Overview
This project is a full-stack, AI-powered defect analytics dashboard built using **Streamlit, Python, and Machine Learning**. It pulls real-time issue data from GitHub repositories and transforms it into meaningful insights using data visualization, predictive modeling, and natural language processing (NLP).

The goal of this project is to simulate how real engineering teams track, analyze, and predict software defects.

---

## 🚀 Key Features

### 🔹 Real-Time Data Pipeline
* Fetches live issue data from multiple GitHub repositories.
* Handles pagination, filtering, and preprocessing.
* Converts raw issue data into structured defect metrics.

### 🔹 Interactive Dashboard
* Filter defects by category, severity, module, and date.
* **Visualizations include:**
    * Defects by severity
    * Time-series trends
    * Heatmaps (Category vs. Module)
* Paginated raw data view for large datasets.

### 🔹 Machine Learning (Regression)
* Predicts **resolution time (in days)** for defects.
* **Features used:**
    * Number of labels
    * Issue title length
    * Bug indicators
* **Model:** Random Forest Regressor

---

## 🧠 NLP-Based Categorization (Core Idea)

### ❓ The Problem
GitHub issues do not come with clean, structured categories like **UI**, **Backend**, or **Performance**. Instead, they contain unstructured text, making manual analysis slow and difficult.

### 🧪 Step 1: TF-IDF + Logistic Regression (Baseline)
We first implemented a lightweight NLP pipeline:
* **TF-IDF:** Converts issue text into numerical features based on word importance.
* **Logistic Regression:** Learns patterns between words and categories.
* *Example:* `"API returns error"` → **Backend**

### ⚠️ Limitation
TF-IDF treats words independently. It doesn't understand that `"login failed"` and `"auth error"` mean the same thing.

### 🔥 Step 2: Embedding-Based NLP (Advanced)
To improve accuracy, we upgraded to **sentence embeddings** using `sentence-transformers`.
* Converts entire sentences into dense vectors.
* Captures **semantic meaning** (not just keywords).
* *Example:* `"login failed"` ≈ `"authentication error"` → Correctly grouped.

---

## 🔍 Explainable AI (XAI)
To make predictions transparent, I used a **cosine similarity** to find similar past issues and display them to justify predictions. This answers: *"Why was this issue classified as Backend?"*

## 🧩 Unsupervised Learning (Clustering)
For issues that don’t fit known categories, we apply **K-Means clustering** to group similar unknown issues together, helping discover hidden patterns in defects.

---

## 🧱 Tech Stack

| Category | Tools |
| :--- | :--- |
| **Frontend** | Streamlit |
| **Data** | Pandas, GitHub Issues API |
| **Visualization** | Plotly |
| **ML Models** | Random Forest, Logistic Regression |
| **NLP** | TF-IDF, Sentence Transformers (BERT-based) |

---

## 📊 Why This Project Matters
This project mimics real-world engineering analytics systems by combining data pipelines, machine learning, and interactive dashboards. It demonstrates how raw, unstructured data can be transformed into actionable insights.

## 🚀 Future Improvements
* Semantic search (natural language query over issues).
* LLM-based chatbot for defect analysis.
* SLA breach prediction.

---

## 💬 Author
**Abhinav Binu** Computer Science @ UT Dallas