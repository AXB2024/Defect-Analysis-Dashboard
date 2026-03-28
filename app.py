import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime
import requests
from dotenv import load_dotenv
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_nlp_model(df):
    # Combine text
    df['text'] = df['title'].fillna('') + " " + df['body'].fillna('')

    # Use existing category as weak labels
    df = df[df['Category'] != 'Other']

    if len(df) < 50:
        return None, None

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    y = df['Category']

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    return model, vectorizer

load_dotenv()

# Cache Data Loading Function
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

"""def map_category(labels):
    names = [l['name'].lower() for l in labels]
    if 'bug' in names:
        return 'Backend'
    elif 'ui' in names:
        return 'UI'
    else:
        return 'Other'
"""

# CATEGORY MAPPING (UPGRADE)
def map_category(labels):
    names = [l['name'].lower() for l in labels]

    if any(x in names for x in ['ui', 'frontend']):
        return 'UI'
    elif any(x in names for x in ['backend', 'api']):
        return 'Backend'
    elif any(x in names for x in ['performance', 'slow']):
        return 'Performance'
    elif any(x in names for x in ['integration', 'compatibility']):
        return 'Integration'
    elif any('bug' in x for x in names):
        return 'Backend'
    else:
        return 'Other'
    
@st.cache_data
def load_data():
    repos = [
        "microsoft/vscode",
        "facebook/react"
        ##"tensorflow/tensorflow"
    ]

    headers = {
        "Accept": "application/vnd.github+json"
    }

    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"


    all_issues = []

    for repo in repos:
        url = f"https://api.github.com/repos/{repo}/issues"

        for page in range(1, 6):  # ~500 per repo
            response = requests.get(url, headers=headers, params={
                "state": "all",
                "per_page": 100,
                "page": page
            })

            data = response.json()
            if not data:
                break

            for issue in data:
                if "pull_request" in issue:
                    continue

                issue["repo"] = repo
                all_issues.append(issue)

    df = pd.DataFrame(all_issues)

    if df.empty:
        return df

    # TRANSFORM DATA
    df['Module'] = df['repo'].apply(lambda x: x.split('/')[1])
    df['Category'] = df['labels'].apply(map_category)

    df['Severity'] = df['labels'].apply(
        lambda x: 'Critical' if any('critical' in l['name'].lower() for l in x)
        else 'High' if any('bug' in l['name'].lower() for l in x)
        else 'Medium'
    )

    df['Reported Date'] = pd.to_datetime(df['created_at'], errors='coerce').dt.tz_localize(None)
    df['Closed Date'] = pd.to_datetime(df['closed_at'], errors='coerce').dt.tz_localize(None)

    df['Resolution Time (days)'] = (
        df['Closed Date'] - df['Reported Date']
    ).dt.days

    df = df.dropna(subset=['Resolution Time (days)'])

    # NLP MODEL
    model, vectorizer = train_nlp_model(df)

    if model is not None:
        df['text'] = df['title'].fillna('') + " " + df['body'].fillna('')
        X_all = vectorizer.transform(df['text'])
        df['NLP Category'] = model.predict(X_all)
    else:
        df['NLP Category'] = df['Category']

    # FEATURE ENGINEERING (ML)
    df['num_labels'] = df['labels'].apply(len)
    df['title_length'] = df['title'].apply(lambda x: len(str(x)))
    df['is_bug'] = df['labels'].apply(
        lambda x: int(any('bug' in l['name'].lower() for l in x))
    )

    # ML MODEL
    features = ['num_labels', 'title_length', 'is_bug']
    target = 'Resolution Time (days)'

    if len(df) > 50:  # avoid crash on small data
        X = df[features]
        y = df[target]

        model = RandomForestRegressor(n_estimators=50)
        model.fit(X, y)

        df['Predicted Resolution Time'] = model.predict(X)
    else:
        df['Predicted Resolution Time'] = df['Resolution Time (days)']

    # Optimize
    for col in ['Category', 'Severity', 'Module']:
        df[col] = df[col].astype('category')

    return df


df = load_data()

# Refresh Button
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Streamlit Page Config
st.set_page_config(page_title="Defect Analysis Dashboard", layout="wide")
st.title("🚨 Defect Analysis Dashboard")

# CATEGORY MODE TOGGLE
category_mode = st.sidebar.radio(
    "Category Type",
    ["Rule-Based", "NLP-Based"]
)

category_column = 'Category' if category_mode == "Rule-Based" else 'NLP Category'

# Sidebar Filters
st.sidebar.header("📊 Filter Defects")

category_filter = st.sidebar.multiselect(
    "Select Category", 
    options=df['Category'].cat.categories.tolist(), 
    default=df['Category'].cat.categories.tolist()
)

severity_filter = st.sidebar.multiselect(
    "Select Severity", 
    options=df['Severity'].cat.categories.tolist(), 
    default=df['Severity'].cat.categories.tolist()
)

module_filter = st.sidebar.multiselect(
    "Select Module", 
    options=df['Module'].cat.categories.tolist(), 
    default=df['Module'].cat.categories.tolist()
)

date_range = st.sidebar.date_input(
    "Select Reported Date Range", 
    [df['Reported Date'].min().date(), df['Reported Date'].max().date()]
)


# Filter Data Function
@st.cache_data
def filter_data(df, categories, severities, modules, date_range, category_column):
    if df.empty:
        return df
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
    
    filtered = df[
        ## (df['Category'].isin(categories)) &
        ## (category_column in df.columns) &
        (df[category_column].isin(categories)) &
        (df['Severity'].isin(severities)) &
        (df['Module'].isin(modules)) &
        (df['Reported Date'] >= start_date) &
        (df['Reported Date'] <= end_date)
    ]
    return filtered

filtered_df = filter_data(df, category_filter, severity_filter, module_filter, date_range, category_column)

# KPI Metrics
st.subheader("📌 Key Metrics")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("Total Defects", len(filtered_df))

kpi2.metric(
    "Avg. Resolution Time (days)", 
    round(filtered_df['Resolution Time (days)'].mean(), 2)
)
kpi3.metric(
    "Critical Defects", 
    len(filtered_df[filtered_df['Severity'] == 'High'])
    ## len(filtered_df[filtered_df['Severity'] == 'Critical'])
)
kpi4.metric(
    "Predicted Resolution",
    round(filtered_df['Predicted Resolution Time'].mean(), 2)
)

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
## heatmap_df = filtered_df.groupby(['Category', 'Module']).size().reset_index(name='Count')
heatmap_df = filtered_df.groupby([category_column, 'Module']).size().reset_index(name='Count')
fig3 = px.density_heatmap(
    heatmap_df, 
    ## x='Category', 
    x = category_column,
    y='Module', 
    z='Count',
    color_continuous_scale='Reds', 
    title="Heatmap of Defects"
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

