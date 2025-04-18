# === mlb_predict_today.py ===
import subprocess
import threading
import requests, re, json, pandas as pd, numpy as np, os, joblib
from datetime import datetime
from itertools import combinations

# === CONFIG ===
tmap = {'KC': 'KCR', 'SD': 'SDP', 'SF': 'SFG', 'TB': 'TBR', 'WAS': 'WSN'}
stats_columns = [
    'R', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'SO', 'BA', 'OBP',
    'pR', 'pH', 'p2B', 'p3B', 'pHR', 'pBB', 'pSO', 'pERA'
]
data_dir = 'data'
flag_file = os.path.join(data_dir, '.pipeline_complete')

# === Run pipeline in a background thread ===
def run_pipeline_in_background():
    def _run():
        print("🔄 Running data pipeline...")
        try:
            subprocess.run(["python", "mlb_data_pipeline.py"], check=True)
            print("✅ Pipeline completed.")
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
        finally:
            with open(flag_file, "w") as f:
                f.write(datetime.now().isoformat())
    threading.Thread(target=_run, daemon=True).start()

# === Utility Functions ===
def scrape_today_games():
    date = datetime.today().strftime("%Y-%m-%d")
    try:
        url = f"https://www.sportsbookreview.com/betting-odds/mlb-baseball/?date={date}"
        r = requests.get(url)
        j = re.findall('__NEXT_DATA__" type="application/json">(.*?)</script>', r.text)
        build_id = json.loads(j[0])['buildId']
        data_url = f"https://www.sportsbookreview.com/_next/data/{build_id}/betting-odds/mlb-baseball/money-line/full-game.json?league=mlb-baseball&oddsType=money-line&oddsScope=full-game&date={date}"
        odds_json = requests.get(data_url).json()
        rows = odds_json['pageProps']['oddsTables'][0]['oddsTableModel']['gameRows']

        games = []
        for g in rows:
            game = {
                'date': g['gameView']['startDate'],
                'home_team_abbr': g['gameView']['homeTeam']['shortName'],
                'away_team_abbr': g['gameView']['awayTeam']['shortName'],
                'home_ml': None,
                'away_ml': None,
                'TmStart': 'TBD',
                'OppStart': 'TBD'
            }
            if g['gameView'].get('homeStarter'):
                s = g['gameView']['homeStarter']
                game['TmStart'] = s.get('firstInital', '') + '.' + s.get('lastName', '')
            if g['gameView'].get('awayStarter'):
                s = g['gameView']['awayStarter']
                game['OppStart'] = s.get('firstInital', '') + '.' + s.get('lastName', '')
            for v in g.get('oddsViews', []):
                if v and v.get('sportsbook', '').lower() == 'fanduel':
                    game['home_ml'] = v['currentLine']['homeOdds']
                    game['away_ml'] = v['currentLine']['awayOdds']
                    break
            games.append(game)
        return pd.DataFrame(games)
    except Exception as e:
        print("Error fetching odds:", e)
        return pd.DataFrame()

def moneyline_to_prob(ml):
    try:
        ml = float(ml)
        return abs(ml) / (abs(ml) + 100) if ml < 0 else 100 / (ml + 100)
    except:
        return np.nan

def kelly_bet(ml, p, bankroll, fraction=0.5):
    try:
        ml = float(ml)
        b = (ml / 100) if ml > 0 else (100 / abs(ml))
        q = 1 - p
        k = ((b * p) - q) / b
        k = max(0, k)
        return round(k * bankroll * fraction, 2)
    except:
        return 0.0

def compute_parlay_ev(combo):
    decimal_odds = np.prod([
        (100 + abs(team['PredictedML'])) / 100 if team['PredictedML'] > 0 else (100 / abs(team['PredictedML']) + 1)
        for team in combo
    ])
    prob_win = np.prod([team['PredictedProb'] for team in combo])
    ev = (decimal_odds * prob_win) - 1
    return decimal_odds, prob_win, ev

# === Main Prediction Logic ===
def run_predictions(bankroll, fraction_kelly):
    model = joblib.load(os.path.join(data_dir, 'rf_mlb_model.joblib'))
    logs = pd.read_csv(os.path.join(data_dir, 'game_logs.csv'))
    logs['Date'] = pd.to_datetime(logs['Date'])

    todays_df = scrape_today_games()
    if todays_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    todays_df['Tm'] = todays_df['home_team_abbr'].replace(tmap)
    todays_df['Opp'] = todays_df['away_team_abbr'].replace(tmap)
    todays_df['Home'] = True
    todays_df['Date'] = pd.to_datetime(todays_df['date']).dt.tz_localize(None).dt.normalize()
    for col in stats_columns:
        todays_df[col] = 0

    df_combined = pd.concat([logs, todays_df], ignore_index=True, sort=False)
    df_combined = df_combined.sort_values(['Tm', 'Date'])

    for col in stats_columns:
        df_combined[f'avg_{col}'] = (
            df_combined.groupby('Tm')[col]
            .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
        )

    df_combined.ffill(inplace=True)
    df_combined.fillna(0, inplace=True)

    latest = df_combined[df_combined['Date'] == pd.Timestamp('today').normalize()].copy()
    latest.rename(columns={'home_ml': 'Tm_ml', 'away_ml': 'Opp_ml'}, inplace=True)

    X_today = latest[['Home','Tm','Opp','TmStart','OppStart','Tm_ml','Opp_ml'] + [f'avg_{col}' for col in stats_columns]]
    pred = model.predict(X_today)
    prob = model.predict_proba(X_today)

    latest['Prediction'] = pred
    latest['Confidence'] = np.max(prob, axis=1)
    latest['Prediction'] = latest['Prediction'].map({1: 'Win', 0: 'Loss'})
    latest['Pred_prob'] = prob[:, 1]
    latest['Tm_prob'] = latest['Tm_ml'].apply(moneyline_to_prob)
    latest['Bet_Edge'] = latest['Pred_prob'] - latest['Tm_prob']

    sorted_latest = latest.sort_values('Bet_Edge', ascending=False)
    sorted_latest['PredictedTeam'] = sorted_latest.apply(
        lambda x: x['Tm'] if x['Pred_prob'] > 0.5 else x['Opp'], axis=1
    )
    sorted_latest['PredictedML'] = sorted_latest.apply(
        lambda x: x['Tm_ml'] if x['Pred_prob'] > 0.5 else x['Opp_ml'], axis=1
    )
    sorted_latest['PredictedProb'] = sorted_latest['Pred_prob']
    sorted_latest['ImpliedProb'] = sorted_latest.apply(
        lambda x: moneyline_to_prob(x['Tm_ml']) if x['Pred_prob'] > 0.5 else moneyline_to_prob(x['Opp_ml']), axis=1
    )
    sorted_latest['PredEdge'] = sorted_latest['PredictedProb'] - sorted_latest['ImpliedProb']
    sorted_latest['Suggested Bet ($)'] = sorted_latest.apply(
        lambda x: kelly_bet(x['PredictedML'], x['PredictedProb'], bankroll, fraction_kelly), axis=1
    )

    underdog_edges = sorted_latest[(sorted_latest['PredictedML'] > 100) & (sorted_latest['PredEdge'] > 0)]

    parlay_combos = []
    for combo in combinations(sorted_latest.to_dict('records'), 3):
        decimal_odds, prob_win, ev = compute_parlay_ev(combo)
        bet_size = round(bankroll * 0.01, 2)  # 1% fixed bet per parlay
        expected_return = round(ev * bet_size, 2)
        parlay_combos.append({
            'Teams': " + ".join(
                f"{g['Tm'] if g['Pred_prob'] > 0.5 else g['Opp']} (vs {g['Opp'] if g['Pred_prob'] > 0.5 else g['Tm']})"
                for g in combo
            ),
            'Decimal Odds': round(decimal_odds, 2),
            'Win Prob': round(prob_win, 4),
            'EV': round(ev, 4),
            'Suggested Bet ($)': bet_size,
            'Expected Return ($)': expected_return
        })

    parlay_df = pd.DataFrame(parlay_combos).sort_values('EV', ascending=False).head(5) if parlay_combos else pd.DataFrame(columns=['Teams', 'Decimal Odds', 'Win Prob', 'EV', 'Suggested Bet ($)', 'Expected Return ($)'])

    return sorted_latest, underdog_edges, parlay_df