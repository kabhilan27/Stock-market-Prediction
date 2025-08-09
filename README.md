# 📈 Stock Market Prediction System

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![yFinance](https://img.shields.io/badge/Data-yFinance-003B57)](https://pypi.org/project/yfinance/)
[![NumPy](https://img.shields.io/badge/Library-NumPy-013243?logo=numpy)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Library-Pandas-150458?logo=pandas)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/ML-TensorFlow-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Matplotlib](https://img.shields.io/badge/Visualization-Matplotlib-11557C?logo=plotly)](https://matplotlib.org/)
[![ReportLab](https://img.shields.io/badge/PDF-ReportLab-00AEEF)](https://www.reportlab.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An **AI-powered stock price prediction system** built with **Python**, **Streamlit**, and **TensorFlow**.  
It fetches real-time stock data, trains an **LSTM model**, visualizes moving averages, and predicts future stock prices.  
Includes professional charts, model saving, and automated PDF report generation. 

---

## ✨ Features

**📊 Stock Data Visualization**
- Fetches **historical and real-time stock price data** from **Yahoo Finance (yFinance)**
- Calculates and displays **50-day (MA50)**, **100-day (MA100)**, and **200-day (MA200)** moving averages
- Interactive stock ticker and date range selection via Streamlit interface
- High-quality charts for professional presentation

**🤖 LSTM Price Prediction**
- Preprocesses stock data into time-series sequences for training
- Builds and trains a **Long Short-Term Memory (LSTM)** neural network for time-series forecasting
- Compares **predicted prices vs actual prices** on a plot
- Saves the trained model (`.keras` format) for reuse without retraining

**📂 Output Files**
- Saves generated **charts** (`.png` format) locally
- Stores the trained **LSTM model**
- Automatically generates a **PDF report** summarizing:
  - Stock information
  - Moving averages
  - Prediction graphs
  - Insights

**⚡ Easy to Use**
- Simple **Streamlit-based UI**
- Just enter:
  - Stock ticker (e.g., `AAPL`, `TSLA`)
  - Date range for historical data
- One click to:
  - Visualize charts
  - Predict future prices
  - Generate and download the report

---

## 🛠 How It Works

1. **Data Fetching**
   - Uses **yFinance API** to retrieve historical stock price data
2. **Data Preprocessing**
   - Scales data using MinMaxScaler
   - Converts price data into LSTM-compatible sequences
3. **Model Training**
   - Builds an LSTM neural network using **TensorFlow**
   - Trains on historical data to detect patterns
4. **Prediction**
   - Uses the trained model to predict stock prices
   - Displays results alongside actual prices
5. **Visualization**
   - Plots moving averages and prediction graphs using Matplotlib
6. **Report Generation**
   - Uses ReportLab to create a PDF report with all results

---

## 🚀 How to Run Locally

1️⃣ **Clone the Repository**
```bash
git clone https://github.com/yourusername/stock-market-prediction.git
cd stock-market-prediction
```

2. **Install Dependencies**

```bash
pip install streamlit yfinance numpy pandas scikit-learn tensorflow matplotlib reportlab
```
   
4. **Run the App**

```bash
streamlit run main.py
```

---

###
