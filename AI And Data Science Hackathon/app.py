

import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Function to train and evaluate models
def train_and_evaluate(X_train, y_train, X_test, model):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

# Main application function

st.title("AI And Data Science Hackathon")
st.subheader('Memon Society')
    
    # Data upload or example data selection
data_source = st.sidebar.file_uploader('Upload dataset', type=['csv', 'xlsx'])
    
    
if data_source is not None:
    # process the uploaded file
        df = pd.read_csv(data_source)  # assuming the uploaded file is in CSV format
        
        
        if not df.empty:
            st.write("Data Head:", df.head())
            st.write("Data Shape:", df.shape)
            st.write("Data Description:", df.describe())
            st.write("Data Info:", df.dtypes)
            st.write("Column Names:", df.columns.tolist())
        
        
        ML_Type = st.sidebar.selectbox("Machine Learning Type", ["Select ML Type", "Supervised", "UnSupervised"])

        if ML_Type == "Supervised":
            model = st.sidebar.selectbox("Select Model", ["Select Model","Logistic Regression","Decision Tree","Random Forest"])
             # Select features and target
            features = st.multiselect("Select features columns", df.columns.tolist())
            target = st.selectbox("Select target column", df.columns.tolist())
            test_size = st.sidebar.slider("Select test split size", 0.1, 0.5, 0.2)
            
            if features and target and model:
                X = df[features]
                y = df[target]
            
        elif ML_Type == "UnSupervised":
            model = st.sidebar.selectbox("Select Model", ["Select Model", "K-Means Clustering", "Hierarchical Clustering","DBSCAN"])
            if model == "K-Means Clustering" or model == "Hierarchical Clustering":
                cluster = st.sidebar.slider("Select Number of Cluster", 2, 10, 1)
            elif model == "DBSCAN":
                eps_d  = st.sidebar.slider("EPS Size", 0.1, 0.5, 0.2)
            
    
            
        # Button to start analysis
        if st.button("Run Analysis"):
            
            if ML_Type == "Supervised":
                
                # Train-test split
            
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                 
                # Initialize model
                if model == 'Logistic Regression':
                    model = LogisticRegression()
                elif model == 'Decision Tree':
                    model = DecisionTreeClassifier()
                    tree=1
                elif model == 'Random Forest':
                    model = RandomForestClassifier()
                
                # Train and evaluate model
                prediction = train_and_evaluate(X_train, y_train, X_test, model)
            
                st.write(f"testing accuracy: {accuracy_score(y_test, prediction)*100:.2f}")
                
                st.subheader("Confusion Matrix")

                cm = confusion_matrix(y_test, prediction)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

                fig, ax = plt.subplots()
                disp.plot(ax=ax, cmap='Blues', colorbar=False)
                st.pyplot(fig)
                
            
            elif ML_Type == "UnSupervised":
                
                # Ensure numeric data only
                df_numeric = df.select_dtypes(include=[np.number]).dropna()

                # If data has more than 2 dimensions, reduce with PCA for visualization
                pca = PCA(n_components=2)
                df_pca = pca.fit_transform(df_numeric)
                df_pca = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'])
                
                
                if model == "K-Means Clustering":
                    
                    kmeans = KMeans(n_clusters=cluster, random_state=42)
                    labels = kmeans.fit_predict(df_numeric)
                    df_pca["Cluster"] = labels
                    st.write(f"**K-Means Clusters:** {np.unique(labels)}")


                    fig, ax = plt.subplots()
                    sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", palette="Set2", data=df_pca, ax=ax)
                    plt.title("K-Means Clustering Visualization")
                    st.pyplot(fig)
                    
                elif model == "Hierarchical Clustering":
                    
                    agg_cluster = AgglomerativeClustering(n_clusters=cluster, metric='euclidean', linkage='ward')
                    labels = agg_cluster.fit_predict(df_numeric)
                    df_pca["Cluster"] = labels
                    st.write(f"**Hierarchical Clusters:** {np.unique(labels)}")

                    fig, ax = plt.subplots()
                    sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", palette="coolwarm", data=df_pca, ax=ax)
                    plt.title("Hierarchical Clustering Visualization")
                    st.pyplot(fig)
                    
                elif model == "DBSCAN":
                    
                    dbscan = DBSCAN(eps=eps_d, min_samples=5)
                    labels = dbscan.fit_predict(df_numeric)
                    df_pca["Cluster"] = labels
                    st.write(f"**DBSCAN Clusters:** {np.unique(labels)}")

                    fig, ax = plt.subplots()
                    sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", palette="Set1", data=df_pca, ax=ax)
                    plt.title("DBSCAN Clustering Visualization")
                    st.pyplot(fig)
                    
    
    
                
st.markdown("""
<style>

/* ===== GLOBAL ===== */
.stApp {
    background: radial-gradient(circle at 30% 30%, #ADD8E6, #0F2027, #1B2735);
    color: #ADD8E6;
    font-family: 'Poppins', sans-serif;
    animation: fadeIn 1.5s ease-in-out;
}


stButton>button {
    background: linear-gradient(90deg, #00FFF0, #007BFF);
    color: black;
    font-weight: 700;
    border-radius: 14px;
    padding: 0.75em 1.5em;
    border: none;
    transition: all 0.3s ease;
    box-shadow: 0 0 20px rgba(0,255,240,0.3);
}
.stButton>button:hover {
    transform: translateY(-3px) scale(1.05);
    box-shadow: 0 0 30px rgba(0,255,240,0.6);
}

div.stButton > button:first-child {
    color: black;
}



</style>
""", unsafe_allow_html=True)