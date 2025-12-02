import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import numpy as np
import copy # Untuk membuat salinan data agar fungsi clustering tidak saling menimpa

# Konfigurasi halaman dan Judul Utama
st.set_page_config(layout="wide")
st.title("💳 Credit Card Customer Segmentation Dashboard")
st.subheader("Model Clustering K-Means & K-Medoids")

# ========================================================
# 1. DATA LOADING & PREPROCESSING (Menggunakan Caching)
# ========================================================

@st.cache_data
def load_and_preprocess_data():
    """Memuat data, menangani missing values, dan melakukan scaling."""
    
    # Perbaikan URL: raw.hithubusercontent.com -> raw.githubusercontent.com
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/ganjar87/data_science_practice/main/CC%20GENERAL.csv", delimiter=",")
    except Exception as e:
        st.error(f"Gagal memuat dataset dari GitHub. Pastikan URL benar. Error: {e}")
        return pd.DataFrame(), np.array([])

    # Perbaikan: df.crop('CUST_ID', axis=1) -> df.drop('CUST_ID', axis=1)
    df_new = df.drop('CUST_ID', axis=1)
    
    # Menghitung median untuk mengisi missing values
    median_min_payments = df_new['MINIMUM_PAYMENTS'].median()
    median_credit_limit = df_new['CREDIT_LIMIT'].median()
    
    # Perbaikan: Penggunaan .fillna() (median() perlu kurung, inplace=True dihapus saat assign)
    df_new['MINIMUM_PAYMENTS'] = df_new['MINIMUM_PAYMENTS'].fillna(median_min_payments)
    df_new['CREDIT_LIMIT'] = df_new['CREDIT_LIMIT'].fillna(median_credit_limit)

    # Scaling Data
    X = df_new.astype(float).values
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    # Mengembalikan DataFrame awal (untuk visualisasi) dan data yang sudah di-scale
    return df_new, X_scaled

df_raw, X_scaled = load_and_preprocess_data()

# ========================================================
# 2. FUNGSI CLUSTERING (K-MEANS)
# ========================================================

def run_kmeans(df_data, X_data):
    """Melakukan clustering K-Means dan menampilkan visualisasi."""
    st.header("K-Means Clustering (k=3)")

    # --- K-Means ---
    k_means = KMeans(n_clusters=3, random_state=42, n_init='auto') # n_init='auto' untuk kompatibilitas
    k_means.fit(X_data)
    labels = k_means.labels_
    df_data['cluster_labels'] = labels

    col1_v, col2_v = st.columns(2)
    
    # Tampilkan Head Data
    with col1_v:
        st.subheader("Data Hasil Clustering")
        st.dataframe(df_data[['PURCHASES', 'PAYMENTS', 'BALANCE', 'cluster_labels']].head())

    # Tampilkan Statistik Cluster
    with col2_v:
        st.subheader("Ukuran Cluster")
        cluster_counts = df_data['cluster_labels'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        st.dataframe(cluster_counts)

    # ========================================================
    # Visualisasi Matplotlib (2D: PURCHASES vs PAYMENTS)
    # ========================================================
    st.subheader("Visualization with Matplotlib (2D)")
    st.markdown("---")

    x1 = df_data["PURCHASES"]
    x2 = df_data["PAYMENTS"]

    fig_matplotlib, ax = plt.subplots(figsize=(10, 8)) # Menggunakan subplots untuk Streamlit
    u_labels = df_data['cluster_labels'].unique()
    
    for i in u_labels:
        ax.scatter(
            x1[df_data['cluster_labels'] == i],
            x2[df_data['cluster_labels'] == i],
            label=f"Cluster {i}"
        )

    ax.set_xlabel('PURCHASES', fontsize=14)
    ax.set_ylabel('PAYMENTS', fontsize=14)
    ax.set_title('K-Means Clustering (PURCHASES vs PAYMENTS)', fontsize=16)
    ax.legend()
    ax.tick_params(labelsize=12)

    st.pyplot(fig_matplotlib)


    # ========================================================
    # Visualization with Seaborn (2D)
    # ========================================================
    st.subheader("Visualization with Seaborn (2D)")
    st.markdown("---")

    fig_seaborn, ax_sns = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        x="PURCHASES",
        y="PAYMENTS",
        hue="cluster_labels",
        data=df_data,
        palette="Paired",
        ax=ax_sns
    )

    plt.legend(loc="lower right")
    st.pyplot(fig_seaborn)

    # ========================================================
    # Visualization with Plotly Express (3D)
    # ========================================================
    st.subheader("Visualization with Plotly Express (3D)")
    st.markdown("---")
    
    # Asumsi kolom 'BALANCE' ada (jika tidak ada, Plotly akan error)
    # Jika kolom 'BALANCE' tidak ada di data Anda, ganti 'BALANCE' dengan kolom lain
    
    fig_plotly = px.scatter_3d(
        df_data,
        x="PURCHASES",
        y="PAYMENTS",
        z="BALANCE", # Menggunakan BALANCE untuk dimensi ketiga
        color="cluster_labels",
        labels={"cluster_labels": "Cluster Labels"},
        height=600
    )
    
    # Mengatur tampilan marker
    fig_plotly.update_traces(marker=dict(size=4))
    
    st.plotly_chart(fig_plotly, use_container_width=True)

# ========================================================
# 3. FUNGSI CLUSTERING (K-MEDOIDS)
# ========================================================

def run_kmedoids(df_data, X_data):
    """Melakukan clustering K-Medoids dan menampilkan visualisasi."""
    st.header("K-Medoids Clustering (k=4)")
    
    # 1. K-Medoids Clustering
    k_medoids = KMedoids(n_clusters=4, random_state=42)
    k_medoids.fit(X_data)
    
    # Mendapatkan label cluster dan menambahkannya ke DataFrame
    labels = k_medoids.labels_
    df_data['cluster_labels'] = labels
    
    col1_v, col2_v = st.columns(2)
    
    # Tampilkan Head Data
    with col1_v:
        st.subheader("Data Hasil Clustering")
        st.dataframe(df_data[['PURCHASES', 'PAYMENTS', 'BALANCE', 'cluster_labels']].head())
        
    # Tampilkan Statistik Cluster
    with col2_v:
        st.subheader("Ukuran Cluster")
        cluster_counts = df_data['cluster_labels'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        st.dataframe(cluster_counts)

    # ========================================================
    # Visualisasi Matplotlib (2D: PURCHASES vs PAYMENTS)
    # ========================================================
    st.subheader("Visualisation with Matplotlib (2D)")
    st.markdown("---")
    
    x1 = df_data['PURCHASES']
    x2 = df_data['PAYMENTS']
    
    fig_matplotlib, ax = plt.subplots(figsize=(10, 8))
    u_labels = np.unique(labels)
    
    # Perbaikan Plotting: Scatter plot per cluster
    for i in u_labels:
        ax.scatter(x1[df_data['cluster_labels'] == i], 
                   x2[df_data['cluster_labels'] == i], 
                   label=f"Cluster {i}") # Menggunakan label i
        
    ax.set_xlabel('PURCHASES', fontsize=14)
    ax.set_ylabel('PAYMENTS', fontsize=14)
    ax.set_title('K-Medoids Clustering (PURCHASES vs PAYMENTS)', fontsize=16)
    ax.legend()
    ax.tick_params(labelsize=12)
    
    st.pyplot(fig_matplotlib)

    # ========================================================
    # Visualisasi dengan Seaborn (2D)
    # ========================================================
    st.subheader("Visualization with Seaborn (2D)")
    st.markdown("---")
    
    fig_seaborn, ax_sns = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        x="PURCHASES",
        y="PAYMENTS",
        hue="cluster_labels",
        data=df_data,
        palette="Paired",
        ax=ax_sns
    )
    
    plt.legend(loc="lower right")
    st.pyplot(fig_seaborn)

    # ========================================================
    # Visualization with Plotly Express (3D)
    # ========================================================
    st.subheader("Visualization with Plotly Express (3D)")
    st.markdown("---")
    
    fig_plotly = px.scatter_3d(
        df_data,
        x="PURCHASES",
        y="PAYMENTS",
        z="BALANCE", 
        color="cluster_labels",
        labels={"cluster_labels": "Cluster Labels"},
        height=600
    )
    
    fig_plotly.update_traces(marker=dict(size=4))
    
    st.plotly_chart(fig_plotly, use_container_width=True)


# ========================================================
# 4. APLIKASI UTAMA (UI/UX)
# ========================================================

# Create sidebar for model selection
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox(
    "Select Clustering Model to Display",
    ("K-Means (k=3)", "K-Medoids (k=4)", "Tampilkan Data Mentah")
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Rows loaded: {len(df_raw)}")
st.sidebar.caption(f"Features: {X_scaled.shape[1]}")


# Main Content Layout (seperti col1 dan col2 di kode Anda, tetapi lebih terstruktur)
st.markdown("## Model Output")
st.markdown("---")

if model_choice == "K-Means (k=3)":
    # Mencegah modifikasi data asli dan data scaled, menggunakan copy()
    run_kmeans(df_raw.copy(), X_scaled.copy())
elif model_choice == "K-Medoids (k=4)":
    # Mencegah modifikasi data asli dan data scaled, menggunakan copy()
    run_kmedoids(df_raw.copy(), X_scaled.copy())
elif model_choice == "Tampilkan Data Mentah":
    st.header("Data Awal (Credit Card General)")
    st.dataframe(df_raw)
    
    # Menampilkan ringkasan Data Cleaning
    st.markdown("### Ringkasan Data Cleaning")
    st.write(f"- Kolom `CUST_ID` telah dihilangkan.")
    st.write(f"- `MINIMUM_PAYMENTS` ({df_raw['MINIMUM_PAYMENTS'].isnull().sum()} missing values) diisi dengan nilai median.")
    st.write(f"- `CREDIT_LIMIT` ({df_raw['CREDIT_LIMIT'].isnull().sum()} missing values) diisi dengan nilai median.")
    st.write(f"- Semua fitur (kecuali `CUST_ID`) telah di-scale menggunakan `StandardScaler`.")


# Tambahkan footer
st.markdown("---")
st.markdown("Aplikasi Streamlit oleh Gemini.")