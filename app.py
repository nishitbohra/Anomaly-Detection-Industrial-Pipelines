!pip install plotly
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# Load the trained model without compiling
MODEL_PATH = r"C:\Users\Kanchan\Downloads\occupancy+detection\anomaly_detection_autoencoder.h5"
model = load_model(MODEL_PATH, compile=False)

# Assuming the model was trained with 50 timesteps
timesteps = 50

def detect_anomalies(data, threshold):
    # Ensure data is a numpy array with the expected shape
    data = np.expand_dims(data, axis=0)
    data = np.expand_dims(data, axis=1)
    
    # Predict reconstruction
    reconstructed = model.predict(data)
    
    # Calculate reconstruction error
    reconstruction_error = np.mean(np.square(data - reconstructed), axis=(1, 2))[0]
    
    # Determine if it's an anomaly
    is_anomaly = reconstruction_error > threshold
    
    return is_anomaly, reconstruction_error

# Streamlit Page Configuration
st.set_page_config(
    page_title="Pipeline Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add a logo/banner
st.image("banner_img.jpg", use_container_width=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
st.sidebar.info(
    """
    This app allows you to:
    - Upload pipeline data as a CSV.
    - Detect anomalies using a pre-trained Autoencoder.
    - Visualize results with dynamic charts.
    """
)
st.sidebar.image("slidebar_img.jpg", caption="Industrial Pipeline", use_container_width=True)

# Main app title and description
st.title("🔍 Anomaly Detection in Industrial Pipelines")
st.write("""
This application uses an **Autoencoder model** to detect anomalies in pipeline time-series data. 
Upload your data to analyze reconstruction errors and identify anomalies.
""")

# File upload section
uploaded_file = st.file_uploader("Upload a CSV file with pipeline data", type=["csv"])

if uploaded_file:
    # Load and display the data
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(data)

    # Ensure the data contains exactly 5 numeric features per row
    if data.shape[1] != 5:
        st.error("The dataset must contain exactly 5 features per row.")
    else:
        # Convert to numpy array
        numeric_data = data.values

        # Detect anomalies for every row
        anomalies = []
        errors = []
        for row in numeric_data:
            is_anomaly, error = detect_anomalies(row, 0)  # Pass a placeholder threshold for now
            anomalies.append(is_anomaly)
            errors.append(error)

        # Calculate the anomaly threshold
        threshold = np.percentile(errors, 95)

        # Detect anomalies again with the calculated threshold
        anomalies = []
        for row in numeric_data:
            is_anomaly, _ = detect_anomalies(row, threshold)
            anomalies.append(is_anomaly)

        # Add results to the dataframe
        data['Reconstruction Error'] = errors
        data['Anomaly Detected'] = [str(a) for a in anomalies]

        # Display results
        st.subheader("Anomaly Detection Results")
        st.write("Below is the updated data with reconstruction errors and anomaly flags:")
        st.dataframe(data[data['Anomaly Detected'] == 'True'])

        # Highlight anomalies in the reconstruction errors graph
        st.subheader("Reconstruction Errors Visualization")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(errors))), y=errors, mode='lines', name='Reconstruction Error'))
        fig.add_trace(
            go.Scatter(
                x=[i for i, a in enumerate(anomalies) if a], 
                y=[errors[i] for i, a in enumerate(anomalies) if a], 
                mode='markers', 
                marker=dict(color='red', size=10), 
                name='Anomalies'
            )
        )
        fig.add_hline(y=threshold, line_dash="dot", line_color="green", annotation_text="Anomaly Threshold", annotation_position="top right")
        fig.update_layout(
            title='Reconstruction Errors with Anomaly Threshold',
            xaxis_title='Index',
            yaxis_title='Reconstruction Error',
        )
        st.plotly_chart(fig, use_container_width=True)

        # Add an explanation of the threshold and reconstruction error
        st.subheader("Understanding Reconstruction Errors")
        st.write("""
        Each point in the above chart represents the reconstruction error for a data point. 
        The green dashed line is the **anomaly threshold**, calculated as the 95th percentile of all reconstruction errors. 
        Any point above this threshold is flagged as an anomaly (red markers).
        """)
        st.markdown("""
        **Why does this happen?**  
        The Autoencoder model attempts to reconstruct normal data effectively, resulting in low reconstruction errors. 
        For anomalies (data that the model hasn’t seen during training or that is significantly different), the reconstruction error is higher, indicating unusual patterns or deviations.
        """)

        # Summary section
        num_anomalies = sum(anomalies)
        st.subheader("Summary")
        st.write(f"Total Rows Processed: **{len(data)}**")
        st.write(f"Anomalies Detected: **{num_anomalies}**")
        st.write(f"Threshold Used: **{threshold:.4f}**")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Developed with ❤️ by Nishit Bohra")
