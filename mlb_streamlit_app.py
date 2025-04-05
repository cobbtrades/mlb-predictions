# mlb_streamlit_app.py
import streamlit as st
import pandas as pd
from mlb_predict_today import run_predictions

# === Streamlit Config ===
st.set_page_config(page_title="MLB Daily Predictions", layout="wide")
st.markdown("<h1 style='text-align: center; color: #e63946;'>⚾ MLB Daily Predictions & Betting Edges</h1>", unsafe_allow_html=True)

# === Run Model Pipeline ===
with st.spinner("🔄 Fetching data and running predictions..."):
    top_edges_df, underdogs_df, parlays_df = run_predictions()

# === Summary Metrics ===
st.markdown("## 📊 Daily Summary")
col1, col2, col3 = st.columns(3)
col1.metric("🧮 Total Games Predicted", len(top_edges_df))
col2.metric("🐶 Underdog Picks", len(underdogs_df))
col3.metric("💰 Top Parlay EV", f"{parlays_df['EV'].max():.2f}" if not parlays_df.empty else "0.00")

st.divider()

# === Styling Helper ===
def render_html_table(df, formats=None, highlight_column=None):
    theme = st.get_option("theme.base") or "light"
    text_color = "white" if theme == "dark" else "black"
    bg_color = "#000" if theme == "dark" else "#fff"

    styles = f"""
    <style>
        table {{
            width: 100%;
            border-collapse: collapse;
            font-family: 'Segoe UI', sans-serif;
            margin-bottom: 1rem;
            text-align: center;
            background-color: {bg_color};
            color: {text_color};
        }}
        thead {{
            background-color: #f1f1f1;
            color: black;
        }}
        th {{
            padding: 12px;
            font-weight: 600;
            border-bottom: 2px solid #ddd;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #eee;
        }}
        tr:hover {{ background-color: #f9f9f9; }}
    </style>
    """

    if formats:
        for col, fmt in formats.items():
            if col in df.columns:
                df[col] = df[col].map(fmt)

    html = styles + "<table><thead><tr>"
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"

    for _, row in df.iterrows():
        html += "<tr>"
        for col in df.columns:
            val = row[col]
            color = ""
            if col == highlight_column:
                if isinstance(val, str) and '%' in val:
                    val_num = float(val.strip('%'))
                    if val_num < 0:
                        color = "red"
                    elif val_num > 0:
                        color = "limegreen"
                elif isinstance(val, (int, float)):
                    if val < 0:
                        color = "red"
                    elif val > 0:
                        color = "limegreen"
            html += f"<td style='color:{color}'>{val}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    return html

# === Top Edges Table ===
st.markdown("### 📈 Top Value Edges (Sorted by Edge)")
if top_edges_df.empty:
    st.info("No predictions available today.")
else:
    edges_df = top_edges_df[[
        'Tm', 'Opp', 'Tm_ml', 'Opp_ml', 'Tm_prob',
        'Pred_prob', 'Bet_Edge', 'Prediction', 'Confidence'
    ]].copy().sort_values("Bet_Edge", ascending=False)

    edges_df.rename(columns={
        'Tm': 'Team', 'Opp': 'Opponent',
        'Tm_ml': 'Team ML', 'Opp_ml': 'Opp ML',
        'Tm_prob': 'Implied Prob', 'Pred_prob': 'Model Prob',
        'Bet_Edge': 'Edge', 'Confidence': 'Conf.'
    }, inplace=True)

    # Compute Winner from predicted probability
    edges_df["Winner"] = edges_df.apply(
        lambda x: x["Team"] if x["Model Prob"] > 0.5 else x["Opponent"],
        axis=1
    )

    edges_df.drop(columns=["Prediction"], inplace=True)

    st.markdown(render_html_table(
        edges_df,
        formats={
            'Team ML': '{:.0f}'.format,
            'Opp ML': '{:.0f}'.format,
            'Implied Prob': '{:.1%}'.format,
            'Model Prob': '{:.1%}'.format,
            'Edge': '{:+.1%}'.format,
            'Conf.': '{:.0%}'.format
        },
        highlight_column="Edge"
    ), unsafe_allow_html=True)

# === Underdogs Table ===
st.markdown("### 🐶 Underdog Value Picks")
if underdogs_df.empty:
    st.info("No strong underdog opportunities found today.")
else:
    dog_df = underdogs_df[[
        'Tm', 'Opp', 'Tm_ml', 'Tm_prob', 'Pred_prob',
        'Bet_Edge', 'Prediction', 'Confidence'
    ]].copy().sort_values("Bet_Edge", ascending=False)

    dog_df.rename(columns={
        'Tm': 'Team', 'Opp': 'Opponent', 'Tm_ml': 'ML',
        'Tm_prob': 'Implied Prob', 'Pred_prob': 'Model Prob',
        'Bet_Edge': 'Edge', 'Confidence': 'Conf.'
    }, inplace=True)

    # Compute Winner from predicted probability
    dog_df["Winner"] = dog_df.apply(
        lambda x: x["Team"] if x["Model Prob"] > 0.5 else x["Opponent"],
        axis=1
    )

    dog_df.drop(columns=["Prediction"], inplace=True)

    st.markdown(render_html_table(
        dog_df,
        formats={
            'ML': '{:.0f}'.format,
            'Implied Prob': '{:.1%}'.format,
            'Model Prob': '{:.1%}'.format,
            'Edge': '{:+.1%}'.format,
            'Conf.': '{:.0%}'.format
        },
        highlight_column='Edge'
    ), unsafe_allow_html=True)

# === Parlays Table ===
st.markdown("### 🎯 Suggested Parlays")
if parlays_df.empty:
    st.info("No parlay recommendations available today.")
else:
    parlay_df = parlays_df[['Teams', 'Decimal Odds', 'Win Prob', 'EV']].copy()
    st.markdown(render_html_table(
        parlay_df,
        formats={
            'Decimal Odds': '{:.2f}'.format,
            'Win Prob': '{:.1%}'.format,
            'EV': '{:+.1%}'.format
        },
        highlight_column='EV'
    ), unsafe_allow_html=True)

# === Footer ===
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with ❤️ using <b>Streamlit</b> & <b>Scikit-learn</b><br>"
    "<i>Data from SportsbookReview & Baseball Reference</i>"
    "</div>",
    unsafe_allow_html=True
)