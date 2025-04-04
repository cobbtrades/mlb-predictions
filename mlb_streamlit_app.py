# mlb_streamlit_app.py

import streamlit as st
import pandas as pd
from mlb_predict_today import run_predictions

# === Streamlit Page Setup ===
st.set_page_config(page_title="MLB Daily Predictions", layout="wide")
st.title("⚾ MLB Daily Predictions & Betting Edges")

# === Run Prediction Pipeline ===
with st.spinner("Running model and fetching data..."):
    top_edges_df, underdogs_df, parlays_df = run_predictions()

# === Summary Metrics ===
st.markdown("---")
st.subheader("📊 Summary")
st.columns(3)[0].metric("Total Games Predicted", len(top_edges_df))
st.columns(3)[1].metric("Underdog Plays", len(underdogs_df))
if not parlays_df.empty:
    top_ev = parlays_df["EV"].max()
    st.columns(3)[2].metric("Top Parlay EV", f"{top_ev:.2f}")
else:
    st.columns(3)[2].metric("Top Parlay EV", "0.00")
st.markdown("---")

# === Top Edges Table ===
st.subheader("📈 Top Betting Edges")
styled_edges = top_edges_df[
    ['Tm', 'Opp', 'Home', 'Tm_ml', 'Opp_ml', 'Tm_prob', 'Pred_prob', 'Bet_Edge', 'Prediction', 'Confidence']
].round(3).style.background_gradient(subset=['Bet_Edge', 'Confidence'], cmap="YlGn")
st.dataframe(styled_edges, use_container_width=True, height=400)

# === Underdog Plays ===
st.subheader("🐶 Underdog Value Plays")
if underdogs_df.empty:
    st.info("No strong underdog opportunities found today.")
else:
    st.dataframe(
        underdogs_df[['Tm', 'Opp', 'Tm_ml', 'Tm_prob', 'Pred_prob', 'Bet_Edge', 'Prediction', 'Confidence']].round(3),
        use_container_width=True
    )

# === Parlay Combos ===
st.subheader("🎯 Suggested Parlays (Top EV Combos)")
if parlays_df.empty:
    st.info("No parlay recommendations available today.")
else:
    st.dataframe(
        parlays_df[['Teams', 'Decimal Odds', 'Win Prob', 'EV']].round(3),
        use_container_width=True
    )

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Scikit-learn | Data from SportsbookReview + Baseball Reference")