# Anomaly-Detection-Industrial-Pipelines
# Anomaly Detection in Industrial Pipelines Using Autoencoder Models

This project demonstrates the application of an **Autoencoder model** to detect anomalies in industrial pipeline time-series data. By analyzing the reconstruction errors of input data, this system can flag unusual patterns or deviations from the normal operational behavior, potentially indicating issues such as malfunctions or failures in the pipeline system.

The goal of this project is to provide an automated, machine learning-driven solution to monitor and maintain pipeline health through anomaly detection.

---

## Key Features

- **CSV File Upload**: Users can upload CSV files containing time-series pipeline sensor data.
- **Anomaly Detection**: The app uses a **pre-trained Autoencoder** model to detect anomalies by analyzing reconstruction errors.
- **Interactive Visualization**: Visualize reconstruction errors and anomalies with interactive plots.
- **Dynamic Threshold Calculation**: Automatically calculates the anomaly detection threshold based on the **95th percentile** of reconstruction errors.

---

## Requirements

To run this project, you need the following Python libraries:

- **Python 3.x**
- **TensorFlow**
- **Streamlit**
- **Pandas**
- **Numpy**
- **Plotly**

---

## Installation

Follow these steps to set up the project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/nishitbohra/Anomaly-Detection-Industrial-Pipelines.git
   cd Anomaly-Detection-Industrial-Pipelines
   
2. **Install the Required Dependencies: Create a virtual environment (optional but recommended) and install the dependencies**:
   ```bash
   pip install -r requirements.txt
   
3. **Run the Streamlit App: Once the dependencies are installed, you can start the application by running**:
    ```bash
       streamlit run app.py
This will launch the app in your browser, where you can upload your pipeline data and start detecting anomalies.

---

## How It Works
1. **Upload Data**: The application accepts CSV files with time-series data containing numerical values representing pipeline sensor readings.

2. **Anomaly Detection**: The Autoencoder model reconstructs the input data, and the reconstruction error is computed. If the reconstruction error exceeds a dynamically calculated threshold (95th percentile of all reconstruction errors), it flags the data as an anomaly.

3. **Visualization**: A dynamic visualization is generated that shows the reconstruction errors over time. Anomalies are marked in red on the plot, and the threshold is indicated by a green dashed line.

4. **Results** : The application displays the data along with the reconstruction errors and flags anomalies with a True or False label. The number of anomalies detected is also summarized.

---

## Model Information

The Autoencoder model used for anomaly detection was trained on time-series data, and it works by reconstructing the input and measuring the reconstruction error. Large reconstruction errors are considered anomalies. This approach is effective for detecting unseen or unusual patterns in pipeline data that differ from the normal operation.

---

## Model Details:

The model was trained with 50 timesteps of data, which is typically the length of the data sequence processed by the Autoencoder.
The anomaly detection threshold is dynamically set based on the 95th percentile of reconstruction errors.

