import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import os

# Load the pre-trained Keras model
model = load_model('C:\\Users\\USER 1\\Documents\\Python\\Stock\\Stock Predictions Model.keras')

# Streamlit UI
st.title('📈 Stock Market Predictor')
st.markdown('Predict future stock prices using a trained deep learning model.')

# User input for stock symbol
stock = st.text_input('Enter Stock Symbol (e.g., AAPL, GOOG, MSFT)', 'GOOG')
start = '2014-01-01'
end = '2024-12-31'

# Download stock data
data = yf.download(stock, start, end, auto_adjust=False)

# Display last few rows of the data
st.subheader('Stock Price Data')
st.dataframe(data.tail())

# Prepare training and test data
data_train = pd.DataFrame(data['Close'][0:int(len(data)*0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data)*0.80):])

# Normalize data for model input
scaler = MinMaxScaler(feature_range=(0, 1))
past_100_days = data_train.tail(100)
data_test_full = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test_full)

# Calculate moving averages
ma_50 = data['Close'].rolling(50).mean()
ma_100 = data['Close'].rolling(100).mean()
ma_200 = data['Close'].rolling(200).mean()

# Save Matplotlib figures as image files
def save_plot(fig, filename):
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)

# Chart 1: Close price vs 50-day MA
st.subheader('Price vs Moving Average of 50 Days')
fig1 = plt.figure(figsize=(10, 5))
plt.plot(data['Close'], label='Close Price')
plt.plot(ma_50, label='MA50', linestyle='--')
plt.legend()
st.pyplot(fig1)
save_plot(fig1, "ma50.png")

# Chart 2: Close price vs 50-day & 100-day MA
st.subheader('Price vs Moving Average of 50 Days & Moving Average of 100 Days')
fig2 = plt.figure(figsize=(10, 5))
plt.plot(data['Close'], label='Close Price')
plt.plot(ma_50, label='MA50', linestyle='--')
plt.plot(ma_100, label='MA100', linestyle='--')
plt.legend()
st.pyplot(fig2)
save_plot(fig2, "ma50_ma100.png")

# Chart 3: Close price vs 100-day & 200-day MA
st.subheader('Price vs Moving Average of 100 Days & Moving Average of 200 Days')
fig3 = plt.figure(figsize=(10, 5))
plt.plot(data['Close'], label='Close Price')
plt.plot(ma_100, label='MA100', linestyle='--')
plt.plot(ma_200, label='MA200', linestyle='--')
plt.legend()
st.pyplot(fig3)
save_plot(fig3, "ma100_ma200.png")

# Prepare test sequences
x_test = []
y_test = []
for i in range(100, data_test_scaled.shape[0]):
    x_test.append(data_test_scaled[i-100:i])
    y_test.append(data_test_scaled[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)

# Make predictions
y_pred = model.predict(x_test)

# Rescale predictions and actual values back to original scale
scale_factor = 1 / scaler.scale_[0]
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

# Chart 4: Actual vs Predicted prices
st.subheader('🔮 Predicted vs Actual Stock Price')
fig4 = plt.figure(figsize=(10, 5))
plt.plot(y_test, label='Actual Price', color='green')
plt.plot(y_pred, label='Predicted Price', color='red')
plt.title(f'{stock} - Predicted vs Actual')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig4)
save_plot(fig4, "pred_vs_actual.png")

# Display option to download predictions as PDF
st.subheader("Download Predictions as PDF")

# Create a DataFrame with predictions
results_df = pd.DataFrame({
    'Actual Price': y_test.flatten(),
    'Predicted Price': y_pred.flatten()
})

# Function to generate a PDF report with predictions and charts
def generate_pdf(df, stock_symbol):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"<b>Stock Predictions for {stock_symbol}</b>", styles['Title']))
    story.append(Spacer(1, 12))

    # Add predictions table
    table_data = [['Index', 'Actual Price', 'Predicted Price']]
    for i, row in df.head(100).iterrows():
        table_data.append([i+1, f"{row['Actual Price']:.2f}", f"{row['Predicted Price']:.2f}"])
    story.append(Table(table_data))
    story.append(Spacer(1, 12))

    # Add saved plots to PDF
    for img_path in ["ma50.png", "ma50_ma100.png", "ma100_ma200.png", "pred_vs_actual.png"]:
        if os.path.exists(img_path):
            story.append(Spacer(1, 12))
            story.append(Image(img_path, width=6*inch, height=3.5*inch))

    doc.build(story)
    buffer.seek(0)
    return buffer

# Generate and offer the PDF file for download
pdf_data = generate_pdf(results_df, stock)

st.download_button(
    label="Download PDF Report with Charts",
    data=pdf_data,
    file_name=f"{stock}_predictions_with_charts.pdf",
    mime="application/pdf"
)
