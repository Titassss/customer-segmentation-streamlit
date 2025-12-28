import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from utils.dataLoad import load_data, load_models
from utils.personaConfig import Personas


st.set_page_config(page_title="Customer Segmentation", layout="wide")

FEATURES = [
    'Income',
    'TotalSpend',
    'Recency',
    'CustomerTenure',
    'NumDealsPurchases',
    'NumWebVisitsMonth',
    'NumWebPurchases',
    'NumCatalogPurchases',
    'NumStorePurchases',
    'TotalChildren'
]

df, pca = load_data()
scaler, kmeans = load_models()

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Customer Profiler"]
)


if page == "Dashboard":
    st.title("Customer Segmentation Dashboard")

 
    st.header(" Executive Overview")

    st.metric("Total Customers", len(df))

    st.subheader("Cluster Distribution")
    dist = df["Cluster"].value_counts(normalize=True) * 100

    for c, pct in dist.items():
        st.write(f"**{Personas[c]['name']}** — {pct:.1f}%")

    st.divider()


    st.header("Cluster Explorer")

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=pca,
        x="PC1", y="PC2",
        hue="Cluster",
        palette="tab10",
        alpha=0.6,
        ax=ax
    )
    st.pyplot(fig)

    cluster = st.selectbox(
        "Select Cluster",
        sorted(df["Cluster"].unique())
    )

    subset = df[df["Cluster"] == cluster]
    persona = Personas[cluster]

    st.subheader(persona["name"])
    st.write(persona["desc"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Spend", int(subset["TotalSpend"].mean()))
    c2.metric("Avg Recency", int(subset["Recency"].mean()))
    c3.metric("Avg Purchases", round(subset["TotalPurchases"].mean(), 1))

    st.subheader("Channel Preference")
    channels = subset[
        ["NumWebPurchases", "NumStorePurchases", "NumCatalogPurchases"]
    ].mean()

    st.bar_chart(channels)

    st.success(f" {persona['action']}")

    st.divider()

    st.header("Customer Personas")

    for p in Personas.values():
        st.markdown(f"""
        ### {p['name']}
        - {p['desc']}
        - **Marketing Action:** {p['action']}
        """)



if page == "Customer Profiler":
    st.title("Individual Customer Profiler")

    with st.form("profile_form"):
        income = st.number_input("Income", 0.0, 200000.0, step=1000.0)
        spend = st.number_input("Total Spend", 0.0, 100000.0, step=500.0)
        recency = st.slider("Recency (days)", 0, 365, 30)
        tenure = st.slider("Customer Tenure (days)", 0, 3000, 365)
        deals = st.slider("Deals Used", 0, 50, 0)
        web_visits = st.slider("Web Visits / Month", 0, 50, 5)
        web = st.slider("Web Purchases", 0, 50, 2)
        catalog = st.slider("Catalog Purchases", 0, 50, 0)
        store = st.slider("Store Purchases", 0, 50, 2)
        children = st.slider("Total Children", 0, 5, 0)

        submit = st.form_submit_button("Profile Customer")

    if submit:
     
        if spend > income:
            st.error("Total Spend cannot exceed Income.")
            st.stop()

        if web + catalog + store == 0:
            st.warning("No purchases detected — results may be unreliable.")

        input_row = {
            'Income': income,
            'TotalSpend': spend,
            'Recency': recency,
            'CustomerTenure': round(tenure/365,2),
            'NumDealsPurchases': deals,
            'NumWebVisitsMonth': web_visits,
            'NumWebPurchases': web,
            'NumCatalogPurchases': catalog,
            'NumStorePurchases': store,
            'TotalChildren': children
        }

       
        X = pd.DataFrame([input_row], columns=FEATURES)
        X_scaled = scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURES)
        cluster = int(kmeans.predict(X_scaled_df)[0])
        persona = Personas[cluster]

        st.success(f"Assigned Segment: {persona['name']}")
        st.write(persona['desc'])
        st.info(f"Recommended Action: {persona['action']}")
