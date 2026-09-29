import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Mobile Price Prediction",
    page_icon="📱",
    layout="wide"
)


# =========================================================
# LOAD DATASET AND MODEL
# =========================================================

df = pd.read_csv("dataset/mobile_prices.csv")

model = joblib.load(
    "model/linear_regression_model.pkl"
)


# =========================================================
# PREPARE DATA FOR MODEL PERFORMANCE
# =========================================================

X = df[
    [
        "RAM",
        "Storage",
        "Battery",
        "ScreenSize",
        "Camera",
        "ProcessorScore"
    ]
]

y = df["Price"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Predictions for test data
y_pred = model.predict(X_test)


# Model metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# =========================================================
# TITLE
# =========================================================

st.title("📱 Mobile Price Prediction")

st.markdown(
    """
    ### Linear Regression Based Machine Learning Application

    Enter the specifications of a mobile phone to estimate
    its price using a trained **Linear Regression model**.
    """
)

st.divider()


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="R² Score",
        value=f"{r2:.3f}"
    )

with col2:
    st.metric(
        label="Mean Absolute Error",
        value=f"₹{mae:,.0f}"
    )

with col3:
    st.metric(
        label="Training Samples",
        value=len(X_train)
    )


st.divider()


# =========================================================
# INPUT SECTION
# =========================================================

st.subheader("📱 Enter Mobile Specifications")


# Two-column layout

col1, col2 = st.columns(2)


with col1:

    ram = st.number_input(
        "RAM (GB)",
        min_value=1,
        max_value=32,
        value=8,
        step=1
    )

    storage = st.number_input(
        "Storage (GB)",
        min_value=16,
        max_value=1024,
        value=128,
        step=16
    )

    battery = st.number_input(
        "Battery (mAh)",
        min_value=1000,
        max_value=10000,
        value=5000,
        step=100
    )


with col2:

    screen_size = st.number_input(
        "Screen Size (inches)",
        min_value=4.0,
        max_value=8.0,
        value=6.5,
        step=0.1
    )

    camera = st.number_input(
        "Camera (MP)",
        min_value=2,
        max_value=250,
        value=50,
        step=1
    )

    processor_score = st.number_input(
        "Processor Score",
        min_value=100,
        max_value=2000,
        value=700,
        step=50
    )


st.divider()


# =========================================================
# PREDICTION
# =========================================================

if st.button(
    "🔮 Predict Mobile Price",
    use_container_width=True
):

    # Create input dataframe

    input_data = pd.DataFrame(
        {
            "RAM": [ram],
            "Storage": [storage],
            "Battery": [battery],
            "ScreenSize": [screen_size],
            "Camera": [camera],
            "ProcessorScore": [processor_score]
        }
    )


    # Make prediction

    prediction = model.predict(
        input_data
    )[0]


    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    st.success("Prediction completed successfully!")

    st.subheader("💰 Predicted Mobile Price")

    st.markdown(
        f"# ₹{prediction:,.2f}"
    )


    # =====================================================
    # MOBILE SPECIFICATION SUMMARY
    # =====================================================

    st.subheader("📋 Mobile Specification Summary")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:

        st.write(f"**RAM:** {ram} GB")
        st.write(f"**Storage:** {storage} GB")
        st.write(f"**Battery:** {battery} mAh")

    with summary_col2:

        st.write(
            f"**Screen Size:** {screen_size} inches"
        )

        st.write(
            f"**Camera:** {camera} MP"
        )

        st.write(
            f"**Processor Score:** {processor_score}"
        )


st.divider()


# =========================================================
# ACTUAL VS PREDICTED GRAPH
# =========================================================

st.subheader("📈 Actual vs Predicted Prices")


chart_data = pd.DataFrame(
    {
        "Actual Price": y_test.values,
        "Predicted Price": y_pred
    }
)


st.line_chart(
    chart_data,
    use_container_width=True
)


st.caption(
    "The chart compares actual test-set prices with "
    "prices predicted by the Linear Regression model."
)


st.divider()

