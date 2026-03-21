# 🩺 FitPulse — Health Anomaly Detection System

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-brightgreen)
![ML](https://img.shields.io/badge/AI-Anomaly%20Detection-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A machine learning system that detects anomalies in health data using 
KMeans clustering and Prophet time-series forecasting, with an 
interactive Streamlit dashboard for visualization.

Built as part of my internship at Infosys Springboard (Sep–Nov 2025).

---

## 🎯 What It Does

- Takes health data (heart rate, sleep duration, daily steps)
- Runs KMeans clustering to group behavioral patterns
- Uses Prophet to model time-series trends and flag deviations
- Displays detected anomalies on an interactive Streamlit dashboard
- Generates risk-level insights based on cluster behavior

---

## 🛠️ Tech Stack

| Component        | Technology              |
|------------------|-------------------------|
| Language         | Python 3.8+             |
| Dashboard        | Streamlit               |
| ML Models        | Scikit-learn, Prophet   |
| Data Processing  | Pandas, NumPy           |
| Visualization    | Matplotlib, Plotly      |

---

## ⚙️ How It Works
```
Health Data (CSV) → Preprocessing → KMeans Clustering
                                  → Prophet Forecasting
                                  ↓
                         Anomaly Detection
                                  ↓
                      Streamlit Dashboard + Risk Insights
```

**KMeans Clustering**
Groups heart rate, sleep, and activity readings into behavioral 
clusters. Points that fall outside expected cluster boundaries 
are flagged as anomalies.

**Prophet Forecasting**
Models the expected trend of each health metric over time. 
Significant deviations from the predicted trend are flagged 
for review.

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/hetpatel1812/FitPulse-Health-Anomaly-Detection.git
cd FitPulse
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run mainapp.py
```

### 4. Upload your data

Upload a CSV file with columns: `heart_rate`, `steps`, `sleep_duration`, `date`

The dashboard will automatically process and display results.

---

## 📁 Project Structure
```
FitPulse/
├── mainapp.py           # Streamlit dashboard entry point
├── requirements.txt     # Python dependencies
├── data/                # Sample health data (CSV)
├── assets/              # Screenshots and diagrams
├── LICENSE
└── README.md
```

---

## 🖥️ Dashboard Features

- Upload CSV health data
- View clustering results across heart rate, sleep, activity
- See flagged anomaly points highlighted on charts
- Download risk insight summary report

---

## 🔮 Planned Improvements

- Add LSTM-based deep learning forecasting
- Connect to real wearable device APIs
- Add real-time alert notifications

---

## 👨‍💻 Developer

**Het Patel**
- GitHub: [hetpatel1812](https://github.com/hetpatel1812)
- LinkedIn: [het-patel-94b334284](https://www.linkedin.com/in/het-patel-94b334284)
- Email: hetpce2005@gmail.com

---

## 📜 License

MIT License — free to use with attribution.

---

## 💬 Feedback

If you find this useful, feel free to star the repo or open an issue.
