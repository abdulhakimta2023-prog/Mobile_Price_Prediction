st.subheader("📈 Actual vs Predicted Prices")

fig, ax = plt.subplots(figsize=(8, 5))

# Scatter plot: Actual prices vs Predicted prices
ax.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

# Reference line: Perfect prediction (y = x)
min_price = min(y_test.min(), y_pred.min())
max_price = max(y_test.max(), y_pred.max())

ax.plot(
    [min_price, max_price],
    [min_price, max_price],
    linestyle="--",
    linewidth=2,
    label="Perfect Prediction"
)

ax.set_xlabel("Actual Price (₹)")
ax.set_ylabel("Predicted Price (₹)")
ax.set_title("Actual vs Predicted Mobile Prices")

ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)

st.caption(
    "Points closer to the dashed reference line indicate predictions "
    "that are closer to the actual mobile prices."
)