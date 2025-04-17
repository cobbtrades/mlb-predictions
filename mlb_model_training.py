# mlb_model_training.py
import pandas as pd, numpy as np, pickle, os, joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime

# === Load and Filter Data ===
df = pd.read_csv('data/game_logs.csv')
odds = pd.read_csv('data/odds_history.csv')

# Convert date columns
df['Date'] = pd.to_datetime(df['Date']).dt.date
odds['date'] = pd.to_datetime(odds['date']).dt.date

# ✅ Filter: only use games up to yesterday
today = datetime.today().date()
df = df[df['Date'] < today]

# === Merge and Feature Engineering ===
odds_home = odds.rename(columns={'home_ml': 'Tm_ml', 'away_ml': 'Opp_ml'})
odds_away = odds.rename(columns={'home_ml': 'Opp_ml', 'away_ml': 'Tm_ml'})

merged_home = pd.merge(df[df['Home']], odds_home, left_on=['Date', 'Tm'], right_on=['date', 'home_team_abbr'], how='left')
merged_away = pd.merge(df[~df['Home']], odds_away, left_on=['Date', 'Tm'], right_on=['date', 'away_team_abbr'], how='left')

df = pd.concat([merged_home, merged_away], ignore_index=True)
df.drop(columns=['date', 'home_team_abbr', 'away_team_abbr'], inplace=True)
df.dropna(inplace=True)

# === Rolling Stats and Averages ===
df = df.sort_values(by=['Date', 'Tm'])
stats_columns = ['R','H','2B','3B','HR','RBI','BB','SO','BA','OBP','pR','pH','p2B','p3B','pHR','pBB','pSO','pERA']

for col in stats_columns:
    df[f'cumsum_{col}'] = df.groupby('Tm')[col].cumsum() - df[col]
    df[f'cumcount_{col}'] = df.groupby('Tm')[col].cumcount()
    df[f'avg_{col}'] = df[f'cumsum_{col}'] / df[f'cumcount_{col}']
    df[f'rolling_{col}_5'] = df.groupby('Tm')[col].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())

df.drop(columns=[f'cumsum_{col}' for col in stats_columns] + [f'cumcount_{col}' for col in stats_columns], inplace=True)
df.bfill(inplace=True)

# === Implied Win Probabilities from Moneylines ===
def moneyline_to_prob(ml):
    try:
        ml = float(ml)
        return -ml / (-ml + 100) if ml < 0 else 100 / (ml + 100)
    except:
        return np.nan

df['Tm_prob'] = df['Tm_ml'].apply(moneyline_to_prob)
df['Opp_prob'] = df['Opp_ml'].apply(moneyline_to_prob)

# === Model Features ===
X = df[['Home','Tm','Opp','TmStart','OppStart','Tm_prob','Opp_prob'] +
       [f'avg_{col}' for col in stats_columns] +
       [f'rolling_{col}_5' for col in stats_columns]]
y = df['Rslt']

# === Preprocessing and Modeling Pipeline ===
categorical_features = ['Tm', 'TmStart', 'Opp', 'OppStart']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, categorical_features)
])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# === Evaluation ===
y_pred = grid_search.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# === Betting Edge Calculation ===
df['model_win_prob'] = grid_search.predict_proba(X)[:, 1]
df['betting_edge'] = df['model_win_prob'] - df['Tm_prob']
df.to_csv('game_logs_with_predictions.csv', index=False)
print("📈 Betting edge and predictions saved to game_logs_with_predictions.csv")

# === Save Model ===
model_path = os.path.join('data', 'rf_mlb_model.joblib')
joblib.dump(grid_search.best_estimator_, model_path, compress=3)
print(f"✅ Compressed model saved to {model_path}")
