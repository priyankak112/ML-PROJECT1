import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.title("Credit Card Customer Segmentation App")


df = pd.read_csv("CC GENERAL.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

df = df.drop('CUST_ID', axis=1)
df.fillna(df.median(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    'PCA_1': X_pca[:, 0],
    'PCA_2': X_pca[:, 1],
    'Cluster': clusters
})

st.subheader("Customer Segmentation (PCA Visualization)")

fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(
    data=pca_df,
    x='PCA_1',
    y='PCA_2',
    hue='Cluster',
    palette='viridis',
    ax=ax
)
st.pyplot(fig)
cluster_names = {
    0: 'High-Risk Revolving Credit Users',
    1: 'Low-Usage / Inactive Customers',
    2: 'High-Value Active Customers',
    3: 'Cash-Dependent Financially Stressed Customers'
}

df['Cluster_Name'] = df['Cluster'].map(cluster_names)
st.subheader("Cluster Summary")
cluster_summary = (
    df.groupby(['Cluster', 'Cluster_Name']).mean().reset_index()
)
st.dataframe(cluster_summary)
