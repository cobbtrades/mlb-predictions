# mlb_data_pipeline.py

from datetime import datetime, timedelta
import os, re, json, time, requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# === CONFIGURATION ===
TEAM_MAP = {'KC': 'KCR', 'SD': 'SDP', 'SF': 'SFG', 'TB': 'TBR', 'WAS': 'WSN'}
TEAMS = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET',
         'HOU', 'KCR', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 'OAK',
         'PHI', 'PIT', 'SDP', 'SEA', 'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSN']
YEARS = [2023, 2024]
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                  " AppleWebKit/537.36 (KHTML, like Gecko)"
                  " Chrome/122.0.0.0 Safari/537.36"
}

# === DATA SCRAPING ===

def fetch_and_process_batting_data(team: str, year: int) -> pd.DataFrame:
    filename = os.path.join(DATA_DIR, f"batting_{team}_{year}.csv")
    if os.path.exists(filename):
        return pd.read_csv(filename, parse_dates=['Date'])

    url = f"https://www.baseball-reference.com/teams/tgl.cgi?team={team}&t=b&year={year}"
    df = pd.read_html(url)[0]
    df = df[df['Opp'] != 'Opp']
    df = df[~df['Date'].str.contains('susp', na=False)]

    df.insert(0, 'Year', year)
    df['Date'] = df['Date'].str.replace(r'\s+\(\d\)', '', regex=True) + ', ' + df['Year'].astype(str)
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')

    df.rename(columns={'Unnamed: 3': 'Home', 'Opp. Starter (GmeSc)': 'OppStart'}, inplace=True)
    df['Home'] = df['Home'].isna()
    df.insert(df.columns.get_loc('Opp'), 'Tm', team)
    df['Rslt'] = df['Rslt'].str.startswith('W').astype(int)
    df['OppStart'] = df['OppStart'].str.replace(r'\(.*\)', '', regex=True)

    df.drop(columns=['Rk', 'Year', '#'], inplace=True)
    numeric_cols = ['PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'IBB', 'SO', 'HBP',
                    'SH', 'SF', 'ROE', 'GDP', 'SB', 'CS', 'LOB', 'BA', 'OBP', 'SLG', 'OPS']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.to_csv(filename, index=False)
    return df

def fetch_and_process_pitching_data(team: str, year: int) -> pd.DataFrame:
    filename = os.path.join(DATA_DIR, f"pitching_{team}_{year}.csv")
    if os.path.exists(filename):
        return pd.read_csv(filename, parse_dates=['Date'])

    url = f"https://www.baseball-reference.com/teams/tgl.cgi?team={team}&t=p&year={year}"
    df = pd.read_html(url)[1]
    df = df[df['Opp'] != 'Opp']
    df = df[~df['Date'].str.contains('susp', na=False)]

    df.insert(0, 'Year', year)
    df['Date'] = df['Date'].str.replace(r'\(\d\)', '', regex=True) + ', ' + df['Year'].astype(str)
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')

    df.insert(df.columns.get_loc('Opp'), 'Tm', team)
    df.rename(columns={df.columns[-1]: 'TmStart'}, inplace=True)
    df['TmStart'] = df['TmStart'].str.split().str[0]

    df.drop(columns=['Year', 'Rk', 'Unnamed: 3', 'Rslt', 'IP', 'BF', '#', 'Umpire'], errors='ignore', inplace=True)
    for col in df.columns:
        if col not in ['Date', 'Opp', 'Tm', 'TmStart']:
            df.rename(columns={col: f'p{col}'}, inplace=True)
            df[f'p{col}'] = pd.to_numeric(df[f'p{col}'], errors='coerce')

    df.to_csv(filename, index=False)
    return df

# === ODDS SCRAPING ===

def daterange(start: datetime.date, end: datetime.date):
    for n in range((end - start).days):
        yield start + timedelta(n)

def fetch_fanduel_mlb_odds(date: datetime.date):
    time.sleep(1.25)
    sport = "mlb-baseball"
    try:
        date_str = date.strftime("%Y-%m-%d")
        url = f"https://www.sportsbookreview.com/betting-odds/{sport}/?date={date_str}"
        r = requests.get(url, headers=HEADERS)

        # Extract the embedded JSON payload
        j = re.findall('__NEXT_DATA__" type="application/json">(.*?)</script>', r.text)
        if not j:
            return []

        parsed = json.loads(j[0])
        build_id = parsed.get("buildId")
        if not build_id:
            return []

        # Construct the JSON URL with the build ID
        data_url = f"https://www.sportsbookreview.com/_next/data/{build_id}/betting-odds/{sport}/money-line/full-game.json?league={sport}&oddsType=money-line&oddsScope=full-game&date={date_str}"
        odds_response = requests.get(data_url, headers=HEADERS)

        if odds_response.status_code != 200:
            return []

        odds_json = odds_response.json()
        page_props = odds_json.get('pageProps')
        if not page_props or 'oddsTables' not in page_props or not page_props['oddsTables']:
            return []

        table_model = page_props['oddsTables'][0].get('oddsTableModel')
        if not table_model or 'gameRows' not in table_model:
            return []

        games = []
        for game in table_model['gameRows']:
            gv = game.get('gameView')
            if not gv:
                continue
            home = gv.get('homeTeam', {}).get('shortName')
            away = gv.get('awayTeam', {}).get('shortName')
            start_date = gv.get('startDate')

            if not all([home, away, start_date]):
                continue

            # Look for FanDuel odds only
            for view in game.get('oddsViews', []):
                if isinstance(view, dict) and view.get('sportsbook', '').lower() == 'fanduel':
                    line = view.get('currentLine', {})
                    games.append({
                        'date': start_date,
                        'home_team_abbr': home,
                        'away_team_abbr': away,
                        'home_ml': line.get('homeOdds', 'N/A'),
                        'away_ml': line.get('awayOdds', 'N/A')
                    })
                    break
        return games

    except Exception as e:
        print(f"[!] Exception on {date_str}: {type(e).__name__}: {e}")
        return []

def fetch_odds_history(start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    odds_file = os.path.join(DATA_DIR, 'odds_history.csv')
    if os.path.exists(odds_file):
        try:
            df = pd.read_csv(odds_file)
            if {'home_team_abbr', 'away_team_abbr', 'date'}.issubset(df.columns):
                return df
        except Exception:
            print("⚠️ odds_history.csv exists but failed to load. Will re-fetch.")

    all_odds = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_fanduel_mlb_odds, date): date for date in daterange(start_date, end_date)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching odds"):
            all_odds.extend(future.result())

    df = pd.DataFrame(all_odds)
    if not df.empty:
        df['home_team_abbr'] = df['home_team_abbr'].replace(TEAM_MAP)
        df['away_team_abbr'] = df['away_team_abbr'].replace(TEAM_MAP)
        df['date'] = pd.to_datetime(df['date']).dt.date
        df.to_csv(odds_file, index=False)
        print(f"✅ Saved odds_history.csv with {len(df)} rows.")
    else:
        print("⚠️ No odds data found.")
    return df

# === PIPELINE EXECUTION ===

def main():
    print("▶ Gathering batting and pitching data...")
    batting_data = pd.concat([fetch_and_process_batting_data(t, y) for y in YEARS for t in TEAMS], ignore_index=True)
    pitching_data = pd.concat([fetch_and_process_pitching_data(t, y) for y in YEARS for t in TEAMS], ignore_index=True)

    # Ensure mergeable game key
    if 'Gtm' in batting_data.columns and 'Gtm' in pitching_data.columns:
        batting_data['Gtm'] = batting_data['Gtm'].astype(str)
        pitching_data['Gtm'] = pitching_data['Gtm'].astype(str)
        merged = pd.merge(batting_data, pitching_data, on=['Gtm', 'Date', 'Tm', 'Opp'])
        merged.to_csv(os.path.join(DATA_DIR, 'game_logs.csv'), index=False)
    else:
        print("❌ Missing merge keys in data.")

    fetch_odds_history(datetime(2023, 3, 31).date(), datetime.now().date())
    print("✅ Data pipeline complete: data/*.csv saved.")

if __name__ == "__main__":
    main()
