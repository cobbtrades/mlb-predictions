# mlb_streamlit_app.py

import streamlit as st
import pandas as pd
from mlb_predict_today import run_predictions

# === Streamlit Page Config ===
st.set_page_config(page_title="MLB Daily Predictions", layout="wide")
st.markdown("<h1 style='text-align: center; color: #e63946;'>⚾ MLB Daily Predictions & Betting Edges</h1>", unsafe_allow_html=True)

# === Run the Backend Model Pipeline ===
with st.spinner("🔄 Fetching data and running predictions..."):
    top_edges_df, underdogs_df, parlays_df = run_predictions()

# === Summary Metrics Display ===
st.markdown("## 📊 Daily Summary")
col1, col2, col3 = st.columns(3)
col1.metric("🧮 Total Games Predicted", len(top_edges_df))
col2.metric("🐶 Underdog Picks", len(underdogs_df))
col3.metric("💰 Top Parlay EV", f"{parlays_df['EV'].max():.2f}" if not parlays_df.empty else "0.00")

st.divider()

# === Top Betting Edges (Cleaned) ===
st.markdown("### 📈 Top Value Edges (Sorted by Edge)")
top_edges_display = top_edges_df[
    ['Tm', 'Opp', 'Home', 'Tm_ml', 'Opp_ml', 'Tm_prob', 'Pred_prob', 'Bet_Edge', 'Prediction', 'Confidence']
].copy()

top_edges_display = top_edges_display.sort_values('Bet_Edge', ascending=False).round(3)

# Apply styling after sorting
styled_edges = top_edges_display.style.background_gradient(
    cmap="YlGn", subset=["Bet_Edge", "Confidence"]
)

st.dataframe(styled_edges, use_container_width=True, hide_index=True, )

# === Underdog Plays ===
st.markdown("### 🐶 Underdog Value Picks")
if underdogs_df.empty:
    st.info("No strong underdog opportunities found today.")
else:
    underdog_display = underdogs_df[
        ['Tm', 'Opp', 'Tm_ml', 'Tm_prob', 'Pred_prob', 'Bet_Edge', 'Prediction', 'Confidence']
    ].copy().sort_values("Bet_Edge", ascending=False).round(3)

    styled_underdogs = underdog_display.style.background_gradient(
        cmap="OrRd", subset=["Bet_Edge"]
    )
    st.dataframe(styled_underdogs, use_container_width=True)

# === Parlay Recommendations ===
st.markdown("### 🎯 Suggested Parlays")
if parlays_df.empty:
    st.info("No parlay recommendations available today.")
else:
    parlays_display = parlays_df[
        ['Teams', 'Decimal Odds', 'Win Prob', 'EV']
    ].copy().sort_values("EV", ascending=False).round(3)

    styled_parlays = parlays_display.style.background_gradient(cmap="BuGn", subset=["EV"])
    st.dataframe(styled_parlays, use_container_width=True)

# === Footer ===
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with ❤️ using <b>Streamlit</b> & <b>Scikit-learn</b><br>"
    "<i>Data from SportsbookReview & Baseball Reference</i>"
    "</div>",
    unsafe_allow_html=True
)
