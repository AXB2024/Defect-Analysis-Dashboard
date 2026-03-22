import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime
import requests
from dotenv import load_dotenv
import os

load_dotenv()

# Cache Data Loading Function
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

@st.cache_data
def load_data():
    repo = "microsoft/vscode"  # you can change this
    url = f"https://api.github.com/repos/{repo}/issues"

    headers = {
        "Accept": "application/vnd.github+json"
    }

    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    params = {
        "state": "all",
        "per_page": 100,
        "page": 1
    }

    all_issues = []

    # Pagination (get up to ~1000 issues)
    for page in range(1, 11):
        params["page"] = page
        response = requests.get(url, headers=headers, params=params)
        data = response.json()

        if not data:
            break

        all_issues.extend(data)

    # Convert to DataFrame
    df = pd.DataFrame(all_issues)

    # Remove pull requests (GitHub mixes them)
    df = df[~df['pull_request'].notna()]

    # Extract fields
    df['Category'] = df['labels'].apply(
        lambda x: x[0]['name'] if x else 'Other'
    )

    df['Severity'] = df['labels'].apply(
        lambda x: 'High' if any('bug' in lbl['name'].lower() for lbl in x) else 'Medium'
    )

    df['Module'] = repo.split('/')[1]

    df['Reported Date'] = pd.to_datetime(df['created_at'], errors='coerce').dt.tz_localize(None)

    df['Closed Date'] = pd.to_datetime(df['closed_at'], errors='coerce').dt.tz_localize(None)

    df['Resolution Time (days)'] = (
        df['Closed Date'] - df['Reported Date']
    ).dt.days

    # Drop rows where issue is still open
    df = df.dropna(subset=['Resolution Time (days)'])

    # Optimize
    for col in ['Category', 'Severity', 'Module']:
        df[col] = df[col].astype('category')

    return df

df = load_data()

# Streamlit Page Config
st.set_page_config(page_title="Defect Analysis Dashboard", layout="wide")
st.title("🚨 Defect Analysis Dashboard")


# Sidebar Filters
st.sidebar.header("📊 Filter Defects")

category_filter = st.sidebar.multiselect(
    "Select Category", options=df['Category'].cat.categories.tolist(), 
    default=df['Category'].cat.categories.tolist()
)

severity_filter = st.sidebar.multiselect(
    "Select Severity", options=df['Severity'].cat.categories.tolist(), 
    default=df['Severity'].cat.categories.tolist()
)

module_filter = st.sidebar.multiselect(
    "Select Module", options=df['Module'].cat.categories.tolist(), 
    default=df['Module'].cat.categories.tolist()
)

date_range = st.sidebar.date_input(
    "Select Reported Date Range", [df['Reported Date'].min().date(), df['Reported Date'].max().date()]
)

# Filter Data Function
@st.cache_data
def filter_data(df, categories, severities, modules, date_range):

    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    if df.empty:
        return df

    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
    
    filtered = df[
        (df['Category'].isin(categories)) &
        (df['Severity'].isin(severities)) &
        (df['Module'].isin(modules)) &
        (df['Reported Date'] >= start_date) &
        (df['Reported Date'] <= end_date)
    ]
    return filtered

filtered_df = filter_data(df, category_filter, severity_filter, module_filter, date_range)

# KPI Metrics
st.subheader("📌 Key Metrics")
kpi1, kpi2, kpi3 = st.columns(3)

kpi1.metric("Total Defects", len(filtered_df))
kpi2.metric("Avg. Resolution Time (days)", round(filtered_df['Resolution Time (days)'].mean(), 2))
kpi3.metric("Critical Defects", len(filtered_df[filtered_df['Severity'] == 'Critical']))

# Visualizations
st.subheader("📈 Visualizations")
col1, col2 = st.columns(2)

# Defects by Severity
with col1:
    severity_count = filtered_df.groupby('Severity').size().reset_index(name='Count')
    fig1 = px.bar(severity_count, x='Severity', y='Count', color='Severity', title="Defects by Severity")
    st.plotly_chart(fig1, use_container_width=True)

# Defects over Time (resampled to daily to optimize plotting)
with col2:
    df_time = filtered_df.set_index('Reported Date').resample('D').size().reset_index(name='Defects')
    fig2 = px.line(df_time, x='Reported Date', y='Defects', title="Defect Trend Over Time")
    st.plotly_chart(fig2, use_container_width=True)

# Heatmap: Category vs Module
st.subheader("🗺️ Heatmap: Category vs Module")
heatmap_df = filtered_df.groupby(['Category', 'Module']).size().reset_index(name='Count')
fig3 = px.density_heatmap(
    heatmap_df, x='Category', y='Module', z='Count',
    color_continuous_scale='Reds', title="Heatmap of Defects"
)
st.plotly_chart(fig3, use_container_width=True)


# Raw Data with Pagination
st.subheader("🧾 Raw Data (Paginated)")
page_size = 500
total_pages = (len(filtered_df) // page_size) + 1
page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
start = (page_number - 1) * page_size
end = start + page_size

st.dataframe(filtered_df.iloc[start:end])

