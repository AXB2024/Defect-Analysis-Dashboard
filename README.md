# Defect Analysis Dashboard

## Overview

The **Defect Analysis Dashboard** is an interactive data visualization tool built with **Python** and **Streamlit** to explore, filter, and analyze software defect data. The dashboard is designed to help users quickly identify defect patterns, severity distribution, resolution efficiency, and risk-prone areas across different modules and categories.

This project focuses on *clarity, usability, and analytical insight* rather than static reporting, allowing stakeholders to explore defects dynamically instead of relying on manual spreadsheets.



## Features

### 🔎 Interactive Filtering

Users can filter defects in real time using:

* **Category** (UI, Backend, Performance, Integration, etc.)
* **Severity** (Critical, High, Medium, Low)
* **Module** (Login, Dashboard, Payments, Reports, Settings)
* **Reported Date Range**

All visualizations and metrics update instantly based on selected filters.



### 📌 Key Metrics (KPIs)

The dashboard surfaces high-level indicators to summarize system health:

* **Total Defects** – number of defects matching selected filters
* **Average Resolution Time** – mean days to resolve defects
* **Critical Defects Count** – immediate risk visibility

These KPIs allow quick assessment before diving into detailed charts.



### 📊 Visual Analytics

#### 1. Defects by Severity (Bar Chart)

Highlights how defects are distributed across severity levels, helping prioritize engineering effort.

#### 2. Defect Trend Over Time (Line Chart)

Displays how defect volume changes over time, making it easier to spot spikes, regressions, or stability periods.

#### 3. Category vs Module Heatmap

Visualizes defect concentration across categories and modules to identify high-risk combinations and recurring problem areas.



### 🧾 Raw Data Exploration

An expandable data table allows users to inspect the filtered dataset directly, supporting transparency and deeper analysis.



## Tech Stack

* **Python** – core programming language
* **Streamlit** – interactive web dashboard framework
* **Pandas** – data manipulation and filtering
* **Plotly Express** – interactive visualizations



## Project Structure

```
├── app.py               # Main Streamlit application
├── requirements.txt    # Project dependencies
└── README.md            # Project documentation
```

---

## How to Run the Dashboard

1. Clone the repository:

```bash
git clone <repository-url>
cd defect-analysis-dashboard
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. Open the local URL provided by Streamlit in your browser.

---

## Data Notes

* The current implementation uses sample defect data generated within the script.
* The dashboard can be easily extended to ingest data from:

  * CSV files
  * Databases
  * Issue tracking systems (e.g., Jira exports)

---

## Future Enhancements

* Integration with real defect tracking tools
