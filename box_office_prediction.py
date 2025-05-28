import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st

# Load Dataset
df = pd.read_csv("movies_metadata.csv")  # Ensure the correct CSV is used

# Data Preprocessing
def preprocess_data(df):
    # Drop rows with missing critical values
    df = df.dropna(subset=['budget', 'runtime', 'popularity', 'vote_average', 'revenue'])

    # Convert necessary columns to numeric
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
    df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')

    df = df.dropna(subset=['budget', 'runtime', 'popularity', 'vote_average', 'revenue'])

    # Log transform revenue and budget
    df['revenue'] = np.log1p(df['revenue'])
    df['budget'] = np.log1p(df['budget'])

    # Encode genres if present
    if 'genres' in df.columns:
        df['genres'] = df['genres'].astype(str).str.slice(2, -2).str.split("'").str[0]
        label_enc = LabelEncoder()
        df['genres'] = label_enc.fit_transform(df['genres'])
    else:
        df['genres'] = 0  # fallback in case no genres

    # Standardize numerical columns
    scaler = StandardScaler()
    numerical_cols = ['budget', 'runtime', 'popularity', 'vote_average']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df, scaler

df, scaler = preprocess_data(df)

# Split Data
X = df[['budget', 'runtime', 'popularity', 'vote_average', 'genres']]
y = df['revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train XGBoost
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Predict and inverse log transformation for evaluation
y_pred_rf = np.expm1(rf_model.predict(X_test))
y_pred_xgb = np.expm1(xgb_model.predict(X_test))
y_test_exp = np.expm1(y_test)

print(f'Random Forest RMSE: {np.sqrt(mean_squared_error(y_test_exp, y_pred_rf))}')
print(f'XGBoost RMSE: {np.sqrt(mean_squared_error(y_test_exp, y_pred_xgb))}')

# Reverse Prediction Model
def revenue_backtracking(expected_revenue, genre):
    expected_revenue_log = np.log1p(expected_revenue)
    sample_data = pd.DataFrame({
        'budget': [np.log1p(expected_revenue * 0.3)],
        'runtime': [(120 + expected_revenue // 50)],
        'popularity': [50],
        'vote_average': [7.0],
        'genres': [genre]
    })
    sample_data[['budget', 'runtime', 'popularity', 'vote_average']] = scaler.transform(
        sample_data[['budget', 'runtime', 'popularity', 'vote_average']]
    )
    return sample_data

# Streamlit Web App
def main():
    st.set_page_config(page_title="🎬 Movie Revenue Predictor", layout="centered")
    st.title("🎬 Movie Box-Office Revenue Predictor")
    st.markdown("Use this tool to **predict revenue** based on your movie features, and get suggestions to hit a target!")

    with st.form("prediction_form"):
        st.subheader("🎥 Movie Features")
        col1, col2 = st.columns(2)
        with col1:
            movie_name = st.text_input("Movie Name")
            budget = st.number_input("Budget (USD)", min_value=1e6, max_value=5e8, step=1e6)
            popularity = st.slider("Popularity Score", min_value=1.0, max_value=100.0, step=1.0)
        with col2:
            runtime = st.slider("Runtime (mins)", min_value=60, max_value=180, step=5)
            vote_average = st.slider("IMDb Rating", min_value=1.0, max_value=10.0, step=0.1)
            genre = st.selectbox("Genre", options=["Action", "Comedy", "Drama", "Sci-Fi", "Horror"])
        genre_map = {"Action": 0, "Comedy": 1, "Drama": 2, "Sci-Fi": 3, "Horror": 4}
        genre_encoded = genre_map[genre]

        submit = st.form_submit_button("🎯 Predict Revenue")

    if submit:
        st.subheader("📈 Predicted Revenue")
        input_df = pd.DataFrame([[np.log1p(budget), runtime, popularity, vote_average, genre_encoded]],
                                columns=X.columns)
        input_df[['budget', 'runtime', 'popularity', 'vote_average']] = scaler.transform(
            input_df[['budget', 'runtime', 'popularity', 'vote_average']]
        )

        rf_log_pred = rf_model.predict(input_df)[0]
        rf_revenue = np.expm1(rf_log_pred)

        xgb_log_pred = xgb_model.predict(input_df)[0]
        xgb_revenue = np.expm1(xgb_log_pred)

        col1, col2 = st.columns(2)
        col1.metric("🌲 Random Forest", f"${rf_revenue:,.2f}")
        col2.metric("🚀 XGBoost", f"${xgb_revenue:,.2f}")

        with st.expander("📊 Compare Predictions"):
            df_results = pd.DataFrame({
                "Model": ["Random Forest", "XGBoost"],
                "Revenue Prediction (USD)": [rf_revenue, xgb_revenue]
            })
            st.dataframe(df_results, use_container_width=True)

    st.divider()
    st.subheader("💡 Budget & Runtime Suggestions")

    col3, col4 = st.columns([2, 1])
    with col3:
        expected_revenue = st.number_input("Target Revenue (USD)", min_value=1e6, max_value=5e8, step=1e6)
    with col4:
        suggest = st.button("🔍 Suggest Budget & Runtime")

    if suggest:
        suggestions = revenue_backtracking(expected_revenue, genre_encoded)
        st.markdown("🎯 **Based on your target, here’s a possible feature set:**")
        st.write(suggestions)


if __name__ == "__main__":
    main()
