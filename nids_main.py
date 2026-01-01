import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI NIDS Dashboard", layout="wide")

st.title("AI-Powered Network Intrusion Detection System")
st.markdown("""
### Project Overview
This system uses Machine Learning (**Random Forest Algorithm**) to analyze network traffic in real-time[cite: 83].
It classifies traffic into two categories:
* **Benign:** Safe, normal traffic[cite: 85].
* **Malicious:** Potential cyberattacks (DDoS, Port Scan, etc.)[cite: 86].
""")

@st.cache_data
def load_data():
    file_path = 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
    try:
        df = pd.read_csv(file_path)
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        df.columns = df.columns.str.strip()
        
        df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
        
        features = ['Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Packet Length Mean', 'Active Mean', 'Label']
        return df[features].head(10000) # Using a subset for faster training
    except FileNotFoundError:
        st.error(f"Error: '{file_path}' not found in the project directory! [cite: 64]")
        return None

df = load_data()
st.sidebar.header("Control Panel")
st.sidebar.info("Adjust model parameters here[cite: 114].")
split_size = st.sidebar.slider("Training Data Size (%)", 50, 90, 80)
n_estimators = st.sidebar.slider("Number of Trees (Random Forest)", 10, 200, 100)

if df is not None:
    X = df.drop('Label', axis=1)
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-split_size)/100, random_state=42)

    st.divider()
    col_train, col_metrics = st.columns([1, 2])

    with col_train:
        st.subheader("1. Model Training")
        if st.button("Train Model Now"):
            with st.spinner("Training Random Forest Classifier... [cite: 127]"):
                model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                model.fit(X_train, y_train)
                st.session_state['model'] = model
                st.success("Training Complete! [cite: 132]")

    if 'model' in st.session_state:
        st.success("Model is Ready for Testing [cite: 134]")

    with col_metrics:
        st.subheader("2. Performance Metrics")
        if 'model' in st.session_state:
            model = st.session_state['model']
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy", f"{acc*100:.2f}%")
            m2.metric("Total Samples", len(df))
            m3.metric("Detected Threats", np.sum(y_pred))
            
            st.write("### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4, 2))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Please train the model first[cite: 153].")

    st.divider()
    st.subheader("3. Live Traffic Simulator (Test the AI)")
    st.write("Enter network packet details below to see if the AI flags it as an attack[cite: 157].")

    c1, c2, c3, c4 = st.columns(4)
    p_dur = c1.number_input("Flow Duration (ms)", 0, 100000, 500)
    p_pkts = c2.number_input("Total Packets", 0, 500, 100)
    p_len = c3.number_input("Packet Length Mean", 0, 1500, 500)
    p_active = c4.number_input("Active Mean Time", 0, 1000, 50)

    if st.button("Analyze Packet"):
        if 'model' in st.session_state:
            model = st.session_state['model']
            input_data = np.array([[80, p_dur, p_pkts, p_len, p_active]])
            pred = model.predict(input_data)
            
            if pred[0] == 1:
                st.error("ALERT: MALICIOUS TRAFFIC DETECTED! [cite: 171]")
                st.write("**Reason:** High packet count with low duration is suspicious[cite: 172].")
            else:
                st.success("Traffic Status: BENIGN (Safe) [cite: 175]")
        else:
            st.error("Please train the model first! [cite: 176]")
else:
    st.warning("Please ensure the CSV file is in the project folder to begin.")