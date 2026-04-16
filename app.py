import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Demand Forecast Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==============================
# LOAD MODEL & COLUMNS
# ==============================
model = joblib.load("demand_forecasting_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# ==============================
# SIDEBAR INPUTS
# ==============================
st.sidebar.title("⚙️ Input Controls")

price = st.sidebar.slider("💰 Price", 0.0, 500.0, 50.0)
discount = st.sidebar.slider("🏷️ Discount", 0.0, 0.9, 0.1)
inventory = st.sidebar.slider("📦 Inventory Level", 0.0, 1000.0, 200.0)

# Optional categorical selections (adjust based on your dataset)
store = st.sidebar.selectbox("🏬 Store ID", ["Store_1", "Store_2", "Store_3"])
category = st.sidebar.selectbox("📦 Category", ["Food", "Clothing", "Electronics"])
region = st.sidebar.selectbox("🌍 Region", ["North", "South", "East", "West"])

predict_btn = st.sidebar.button("🔮 Predict Demand")

# ==============================
# HEADER
# ==============================
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>📦 Inventory Demand Dashboard</h1>",
    unsafe_allow_html=True
)

st.markdown("---")

# ==============================
# KPI SECTION
# ==============================
kpi1, kpi2, kpi3 = st.columns(3)

if predict_btn:

    # ==============================
    # CREATE INPUT DATAFRAME
    # ==============================
    input_data = pd.DataFrame(columns=model_columns)

    # Fill with zeros
    input_data.loc[0] = 0

    # Fill numeric values
    if 'Price' in input_data.columns:
        input_data['Price'] = price

    if 'Discount' in input_data.columns:
        input_data['Discount'] = discount

    if 'Inventory Level' in input_data.columns:
        input_data['Inventory Level'] = inventory

    # ==============================
    # HANDLE CATEGORICAL VALUES
    # ==============================

    # Store
    store_col = f"Store ID_{store}"
    if store_col in input_data.columns:
        input_data[store_col] = 1

    # Category
    cat_col = f"Category_{category}"
    if cat_col in input_data.columns:
        input_data[cat_col] = 1

    # Region
    region_col = f"Region_{region}"
    if region_col in input_data.columns:
        input_data[region_col] = 1

    # ==============================
    # PREDICTION
    # ==============================
    prediction = model.predict(input_data)[0]

    # ==============================
    # KPI DISPLAY
    # ==============================
    kpi1.metric("📊 Predicted Demand", f"{prediction:.2f}")
    kpi2.metric("💰 Price", f"{price}")
    kpi3.metric("📦 Inventory", f"{inventory}")

    st.markdown("---")

    # ==============================
    # VISUALIZATION SECTION
    # ==============================
    col1, col2 = st.columns(2)

    # Bar Chart
    with col1:
        st.subheader("📊 Feature Comparison")

        features = ["Price", "Discount", "Inventory", "Demand"]
        values = [price, discount, inventory, prediction]

        fig, ax = plt.subplots()
        ax.bar(features, values)
        ax.set_title("Feature Contribution")
        st.pyplot(fig)

    # Trend Simulation
    with col2:
        st.subheader("📈 Demand Trend")

        trend = prediction + np.random.randn(10) * 5

        fig2, ax2 = plt.subplots()
        ax2.plot(trend, marker='o')
        ax2.set_title("Future Demand Trend")
        st.pyplot(fig2)

    st.markdown("---")

    # ==============================
    # DEMAND LEVEL INSIGHT
    # ==============================
    st.subheader("📌 Demand Insight")

    if prediction > 100:
        st.success("🔥 High Demand — Increase Stock!")
    elif prediction > 50:
        st.warning("⚡ Moderate Demand — Monitor Closely")
    else:
        st.error("📉 Low Demand — Reduce Inventory")

else:
    st.info("👈 Use sidebar to enter values and click Predict")

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown(
    "<center>🚀 Built with Streamlit | ML Demand Forecasting Project</center>",
    unsafe_allow_html=True
)