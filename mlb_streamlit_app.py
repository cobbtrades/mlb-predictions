import streamlit as st
import pandas as pd
from mlb_predict_today import run_predictions

# === Page Config ===
st.set_page_config(page_title="MLB Daily Predictions", layout="wide")
st.title("⚾ MLB Daily Predictions & Betting Edges")

# === Run Prediction Logic ===
with st.spinner("🔄 Running model and fetching data..."):
    top_edges_df, underdogs_df, parlays_df = run_predictions()

# === Handle No Games Case ===
if top_edges_df.empty:
    st.warning("No games available for prediction today.")
    st.stop()

# === KPI Metrics ===
total_games = len(top_edges_df)
underdog_count = len(underdogs_df)
best_ev = parlays_df["EV"].max() if not parlays_df.empty else 0

col1, col2, col3 = st.columns(3)
col1.metric("📊 Total Games Predicted", f"{total_games}")
col2.metric("🐶 Underdog Plays", f"{underdog_count}")
col3.metric("💰 Top Parlay EV", f"{best_ev:.2f}")

# === Top Betting Edges ===
st.subheader("📈 Top Betting Edges")
styled_edges = top_edges_df[['Tm', 'Opp', 'Home', 'Tm_ml', 'Opp_ml', 'Tm_prob', 'Pred_prob', 'Bet_Edge', 'Prediction', 'Confidence']].round(3)
styled_edges = styled_edges.style.background_gradient(
    subset=['Bet_Edge'], cmap='Greens'
).format({
    'Tm_prob': '{:.2%}',
    'Pred_prob': '{:.2%}',
    'Confidence': '{:.2%}',
    'Bet_Edge': '{:+.3f}'
})
st.dataframe(styled_edges, use_container_width=True, height=400)

# === Underdog Picks ===
st.subheader("🐶 Underdog Value Picks")
if underdogs_df.empty:
    st.info("No strong underdog opportunities today.")
else:
    styled_underdogs = underdogs_df[['Tm', 'Opp', 'Tm_ml', 'Tm_prob', 'Pred_prob', 'Bet_Edge', 'Prediction', 'Confidence']].round(3)
    styled_underdogs = styled_underdogs.style.background_gradient(
        subset=['Bet_Edge'], cmap='Oranges'
    ).format({
        'Tm_prob': '{:.2%}',
        'Pred_prob': '{:.2%}',
        'Confidence': '{:.2%}',
        'Bet_Edge': '{:+.3f}'
    })
    st.dataframe(styled_underdogs, use_container_width=True, height=350)

# === Parlay Suggestions ===
st.subheader("🎯 Suggested Parlays (Top EV Combos)")
if parlays_df.empty:
    st.info("No parlay recommendations available today.")
else:
    parlays_display = parlays_df[['Teams', 'Decimal Odds', 'Win Prob', 'EV']].copy()
    parlays_display['Win Prob'] = parlays_display['Win Prob'].map('{:.2%}'.format)
    parlays_display['EV'] = parlays_display['EV'].map('{:+.2f}'.format)
    st.dataframe(parlays_display, use_container_width=True, height=300)

# === Footer ===
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Scikit-learn | Data from SportsbookReview + Baseball Reference")
