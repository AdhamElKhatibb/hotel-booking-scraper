import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud

st.set_page_config(page_title="Hotel Pricing Dashboard", layout="wide")

menu = st.sidebar.radio("Navigation", [
    "🏠 Home", 
    "📊 Data Overview", 
    "📈 Visualizations", 
    "🤖 Machine Learning Insights", 
    "🧹 NLP Cleaning Process", 
    "📁 Raw Data Preview"
])

@st.cache_data
def load_data():
    try:
        return pd.read_csv("nyc_hotels_data_partial_45000.csv")

    except:
        return None

df = load_data()

@st.cache_data
def load_nlp_data():
    try:
        return pd.read_csv("nlp_reviews_cleaned.csv")
    except:
        return None

nlp_df = load_nlp_data()

if menu == "🏠 Home":
    st.title("Hotel Pricing Analysis Dashboard")
    st.markdown("""
        Welcome to the **Hotel Pricing Analysis** project.

        This dashboard showcases insights extracted from over **45,000 hotels scraped from Booking.com**. 
        Using advanced **NLP techniques**, thorough data cleaning, and **machine learning models**, we:

        - Analyzed key hotel features and pricing
        - Visualized patterns and trends
        - Built predictive models for pricing

        Use the sidebar to navigate through the different insights and data visualizations.
    """)

elif menu == "📊 Data Overview":
    st.title("Data Overview")
    if df is not None:
        st.write("### Columns in Data")
        st.write(df.columns)

        st.write("### Dataset Summary")
        st.dataframe(df.describe())

        st.write("### Missing Values")
        st.dataframe(df.isnull().sum())
    else:
        st.warning("Data not loaded.")

elif menu == "📈 Visualizations":
    st.title("Data Visualizations")
    if df is not None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        if len(numeric_cols) > 1:
            st.write("### Correlation Heatmap")
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        if 'price' in df.columns:
            st.write("### Price Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df['price'], bins=50, kde=True, ax=ax)
            st.pyplot(fig)

        if all(col in df.columns for col in ['numeric_review_count', 'numeric_score', 'price']):
            st.write("### Distribution of Reviews, Ratings, and Prices")
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))

            sns.histplot(df['numeric_review_count'].dropna(), kde=True, color='blue', ax=axs[0])
            axs[0].set_title('Number of Reviews')

            sns.histplot(df['numeric_score'].dropna(), kde=True, color='green', ax=axs[1])
            axs[1].set_title('Hotel Ratings')

            sns.histplot(df['price'].dropna(), kde=True, color='red', ax=axs[2])
            axs[2].set_title('Hotel Prices')

            st.pyplot(fig)
    else:
        st.warning("Data not available.")

elif menu == "🤖 Machine Learning Insights":
    st.title("Machine Learning Insights")
    st.markdown("Model results predicting hotel ratings using price, distance, stars, and reviews.")

    st.subheader("Model Performance (Sample Values)")
    st.metric("R-squared (R²)", "0.72")
    st.metric("Mean Absolute Error (MAE)", "0.35")
    st.metric("Root Mean Squared Error (RMSE)", "0.49")

    try:
        preds = pd.read_csv("predicted_scores.csv")
        
        st.subheader("Predicted vs Actual Ratings")
        fig1, ax1 = plt.subplots()
        sns.scatterplot(data=preds, x='Actual', y='Predicted', ax=ax1)
        ax1.plot([preds['Actual'].min(), preds['Actual'].max()], [preds['Actual'].min(), preds['Actual'].max()], 'r--')
        ax1.set_title("Predicted vs Actual Hotel Scores")
        st.pyplot(fig1)

        st.subheader("Prediction Error Distribution")
        preds['residual'] = preds['Actual'] - preds['Predicted']
        fig2, ax2 = plt.subplots()
        sns.histplot(preds['residual'], bins=30, kde=True, ax=ax2, color='orange')
        ax2.set_title("Distribution of Residuals (Errors)")
        st.pyplot(fig2)

        st.subheader("Residual Plot - Linear Regression")
        residuals = preds['Actual'] - preds['Predicted']
        fig_resid, ax_resid = plt.subplots(figsize=(10, 6))
        sns.histplot(residuals, kde=True, bins=50, ax=ax_resid)
        ax_resid.axvline(0, color="red", linestyle="--")
        ax_resid.set_xlabel("Residual (Error)")
        ax_resid.set_ylabel("Frequency")
        ax_resid.set_title("Residual Plot - Linear Regression")
        st.pyplot(fig_resid)

        st.subheader("Model Comparison Table")
        model_comparison = pd.DataFrame({
            'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
            'R²': [0.72, 0.84, 0.86],
            'MAE': [0.35, 0.28, 0.25],
            'RMSE': [0.49, 0.36, 0.32]
        })
        st.dataframe(model_comparison)

        if 'Actual' in preds.columns and 'Predicted' in preds.columns:
            st.subheader("Classification Report & Confusion Matrix")
            y_true = preds['Actual'].round().astype(int)
            y_pred = preds['Predicted'].round().astype(int)
            report = classification_report(y_true, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

            cm = confusion_matrix(y_true, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            ax_cm.set_title("Confusion Matrix")
            st.pyplot(fig_cm)

        st.download_button(
            label="📥 Download Predictions CSV",
            data=preds.to_csv(index=False).encode('utf-8'),
            file_name='predicted_scores.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.warning("Prediction file not found or unreadable.")
        st.text(f"Debug info: {e}")

elif menu == "🧹 NLP Cleaning Process":
    st.title("NLP Cleaning Process")
    st.markdown("""
        Shows how hotel name text was cleaned and classified using NLP:
        - Tokenization
        - Stopword removal
        - Stemming / Lemmatization
        - TF-IDF & Classification
    """)

    try:
        if nlp_df is not None:
            st.write("**Original Hotel Name:**")
            st.info(nlp_df['name'].iloc[0])
            st.write("**After NLP Processing:**")
            st.success(nlp_df['processed_text'].iloc[0])

            st.write("### Word Cloud from Processed Hotel Names")
            text = " ".join(nlp_df['processed_text'].dropna().astype(str).tolist())
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)

            st.markdown("### Sample Classification Report")
            st.code("""
            precision    recall  f1-score   support
            Excellent       1.00      0.98      0.99      1666
            Good            1.00      1.00      1.00      5120
            Very Good       0.99      1.00      1.00      5842
            accuracy                            1.00     12628
            macro avg       1.00      0.99      0.99     12628
            weighted avg    1.00      1.00      1.00     12628
            """)
        else:
            st.warning("NLP CSV file not found or invalid.")
    except Exception as e:
        st.warning("NLP data could not be displayed.")
        st.text(f"Error: {e}")

elif menu == "📁 Raw Data Preview":
    st.title("Raw Data Preview")
    if df is not None:
        st.dataframe(df.head(100))
    else:
        st.warning("No data loaded.")
