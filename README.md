# Predicting Highway Maintenance and Deterioration Trends

Welcome to our capstone project on predicting highway maintenance and deterioration trends! This project leverages data fusion, machine learning, and time series modeling techniques to forecast pavement deterioration using publicly available transportation datasets.

---

## 🚧 Project Overview

Road maintenance decisions are often reactive due to a lack of predictive insights. This project aims to provide data-driven forecasts of road deterioration to support **proactive maintenance planning and resource optimization**.

We fused data from the **Highway Performance Monitoring System (HPMS)** and the **Freight Analysis Framework (FAF)** (2013–2022), engineered features, and applied time series models to predict pavement deterioration, measured via the **International Roughness Index (IRI)** — a critical metric for assessing road surface quality.

---

## 📊 Data Description

The project uses a cleaned and processed version of longitudinal transportation data sourced from HPMS and FAF. Key features include:

- `ROUTE_ID`: Unique identifier for each highway route  
- `BEGIN_POIN`, `END_POIN`: Starting and ending mile markers  
- `AADT`: Annual Average Daily Traffic  
- `IRI_VN`: International Roughness Index (target variable)  
- `VEHICLE_TONS`, `TRUCK_PCT`, `ROAD_CURVE_DEGREE`  
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
