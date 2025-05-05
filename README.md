# Predicting Highway Maintenance and Deterioration Trends

Welcome to our capstone project on predicting highway maintenance and deterioration trends! This project leverages data fusion, machine learning, and time series modeling techniques to forecast pavement deterioration using publicly available transportation datasets.

---

## 🚧 Project Overview

Road maintenance decisions are often reactive due to a lack of predictive insights. This project aims to provide data-driven forecasts of road deterioration to support **proactive maintenance planning and resource optimization**.

We fused data from the **Highway Performance Monitoring System (HPMS)** and the **Freight Analysis Framework (FAF)** (2013–2022), engineered features, and applied time series models to predict pavement deterioration, measured via the **International Roughness Index (IRI)** — a critical metric for assessing road surface quality.

---

## 🔗 Data Sources

The data used in this project comes from publicly available transportation datasets:

- 📦 [Freight Analysis Framework (FAF) – Bureau of Transportation Statistics](https://www.bts.gov/faf)
- 🛣️ [Highway Performance Monitoring System (HPMS) – Federal Highway Administration](https://www.fhwa.dot.gov/policyinformation/hpms.cfm)

These sources provide essential information on freight movement, roadway usage, traffic metrics, and pavement conditions across the U.S.

---

## 📊 Data Description

The project uses combined transportation datasets derived from the **Highway Performance Monitoring System (HPMS)** and the **Freight Analysis Framework (FAF)** for the years 2013 to 2022. Below are the key features used in the modeling process:

| Feature         | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `dms_orig`      | Origin FAF region (where freight movement begins)                           |
| `dms_dest`      | Destination FAF region (where freight movement ends)                        |
| `dms_mode`      | Mode of Transport (e.g., Truck, Rail, Air, Water)                           |
| `curval`        | Current freight value on the segment                                        |
| `tmiles`        | Ton-miles (freight volume × distance traveled)                              |
| `tons`          | Freight volume (in tons)                                                    |
| `value`         | Monetary value of the freight transported                                   |
| `IRI_VN`        | International Roughness Index (used as the target variable)                |
| `AADT_VN`       | Annual Average Daily Traffic (vehicle volume)                              |
| `IS_IMPROVED`   | Flag indicating if the segment was improved since the previous year         |
| `THROUGH_LA`    | Number of through lanes on the road segment                                 |
| `SPEED_LIMI`    | Posted speed limit for the road segment                                     |

These features were selected for their relevance to pavement deterioration and freight movement patterns across the national highway system.

- Year range: **2013 to 2022**

The dataset was preprocessed to align measurements over time and enable time series modeling at a per-section level.

---

## 📦 Requirements

Before running the project, make sure the following are installed:

- Python 3.10+
- Google Cloud SDK (if deploying to Cloud Run)

### Python dependencies (install via `requirements.txt`):

```bash
pip install -r requirements.txt
```

---

## 🛠️ Setup Instructions

Follow these steps to clone and run the project locally or deploy it to Google Cloud Run.

### 🔁 Clone the repository

```bash
git clone https://github.com/your-username/highway-iri-prediction.git
cd highway-iri-prediction
```

### 🧪 Run locally

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask app:
   ```bash
   python app.py
   ```

4. Open your browser at `http://localhost:5000/home`

---

## 🧠 Libraries Used

- **Flask** – web framework for app routing  
- **Pandas, NumPy** – data manipulation and transformation  
- **Matplotlib, Seaborn** – plotting and visualization  
- **Folium** – interactive mapping of highway segments  
- **TensorFlow / Keras** – time series modeling using TCN  
- **Joblib** – model serialization  
- **Gunicorn** – production server for deployment  

---

## 🧮 Models Used

We trained and deployed a **Temporal Convolutional Network (TCN)** using TensorFlow/Keras to model and predict IRI trends for specific road segments.

Key modeling steps:
- Feature standardization using `StandardScaler`  
- Time series window generation for sequence input  
- Multi-step regression using TCN  
- Deployment of a trained `.h5` model with preloaded scalers  

---

## 🌐 Live Demo

View the live deployed application on Google Cloud Run here:  
🔗 **[Live IRI Prediction App](https://flask-app-213433359152.us-central1.run.app)**

---

## Demo Screenshots 
On the home page you find information about our project such as introduction, data description, and usage.
![image](https://github.com/user-attachments/assets/b950df7e-0167-4b51-83f0-23929dd8f6ff)

once user clicks on **Predict IRI**, user will be redirected to Prediction page.
![image](https://github.com/user-attachments/assets/2ad4c0b1-9797-4b34-9183-061fbe7f34e5)


On the above page user can select a route from the route selection dropdown.
And section or sections from the section selection.
![image](https://github.com/user-attachments/assets/c2b5402e-95e7-4419-ac4e-b142dbd0f522)

and then click on predict

![image](https://github.com/user-attachments/assets/c00904a1-bd0e-4e03-964e-3e1cdd073d22)

for more insights on specfic segements user can select from dropdown








## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository  
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and commit:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to your forked repo:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a Pull Request on GitHub

---

## 📧 Contact

For questions or suggestions, feel free to contact us:

- ✉️ **Chandra Sekhar Katipalli** – gr50572@umbc.com  
- ✉️ **Sindura Reddy Challa** – schalla6@umbc.com  
- ✉️ **Sanjana Reddy Soma** – wl18137@umbc.edu.com  

---

## 👩‍💻 Authors

This project was developed as part of our capstone for the Master’s in Data Science program.

- **Chandra Sekhar Katipalli**  
- **Sindura Reddy Challa**  
- **Sanjana Reddy Soma**
