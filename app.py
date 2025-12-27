import streamlit as st
import pandas as pd
import json
import altair as alt

# 1. Page Configuration
st.set_page_config(page_title="Brand Reputation Monitor 2023", layout="wide")

# 2. Data Loading Function
@st.cache_data
def load_data():
    # Load Products and Testimonials from JSON
    with open("scraped_data.json", "r") as f:
        raw_data = json.load(f)
    
    df_prod = pd.DataFrame(raw_data["products"])
    df_prod["type"] = "product"
    
    df_test = pd.DataFrame(raw_data["testimonials"])
    df_test["type"] = "testimonial"
    
    # Load PRE-ANALYZED reviews from CSV
    df_rev = pd.read_csv("analyzed_reviews.csv")
    df_rev["type"] = "review"
    
    # Combine
    df_final = pd.concat([df_prod, df_test, df_rev], axis=0, ignore_index=True, sort=False)
    df_final['date'] = pd.to_datetime(df_final['date'])
    
    return df_final

try:
    df = load_data()
except FileNotFoundError:
    st.error("Missing files! Ensure 'scraped_data.json' AND 'analyzed_reviews.csv' are on GitHub.")
    st.stop()

# 3. Sidebar Navigation
st.sidebar.title("Brand Monitor 2023")
page = st.sidebar.radio("Navigate to:", ["Products", "Testimonials", "Reviews"])

# 4. Sections
if page == "Products":
    st.header("📦 Product Catalog")
    products_df = df[df['type'] == 'product'].copy()
    st.dataframe(products_df[["name", "price", "short-description"]], use_container_width=True, hide_index=True)

elif page == "Testimonials":
    st.header("💬 Customer Testimonials")
    testimonials_df = df[df['type'] == 'testimonial'].copy()
    st.dataframe(testimonials_df[["text", "rating"]].dropna(subset=["text"]), use_container_width=True, hide_index=True)

elif page == "Reviews":
    st.header("📊 2023 Review Sentiment Analysis")

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    selected_month_name = st.select_slider("Filter reviews by month (2023):", options=months, value="May")
    month_num = months.index(selected_month_name) + 1
    
    filtered_reviews = df[
        (df['type'] == 'review') & 
        (df['date'].dt.month == month_num) & 
        (df['date'].dt.year == 2023)
    ].copy()

    if filtered_reviews.empty:
        st.warning(f"No reviews found for {selected_month_name} 2023.")
    else:
        st.write(f"Found **{len(filtered_reviews)}** reviews for {selected_month_name}:")
        
        # Display the reviews with sentiment results already there!
        st.divider()
        st.subheader("Sentiment Analysis Results")

        def color_sentiment(val):
            return f'background-color: {"#90ee90" if val == "POSITIVE" else "#ffcccb"}'

        # Format confidence for display
        filtered_reviews["confidence"] = filtered_reviews["score_val"].apply(lambda x: f"{x:.2%}")

        styled_df = filtered_reviews[["date", "text", "sentiment", "confidence"]].style.applymap(
            color_sentiment, subset=['sentiment']
        )
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Metrics
        col1, col2 = st.columns(2)
        col1.metric("Positive Reviews", (filtered_reviews["sentiment"] == "POSITIVE").sum())
        col2.metric("Negative Reviews", (filtered_reviews["sentiment"] == "NEGATIVE").sum())

        # Visualization
        st.divider()
        st.subheader(f"Sentiment Distribution: {selected_month_name}")

        chart_df = filtered_reviews.groupby("sentiment")["score_val"].agg(['count', 'mean']).reset_index()
        chart_df.columns = ["Sentiment", "Count", "Average Confidence"]

        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X('Sentiment:N', title="Sentiment Type"),
            y=alt.Y('Count:Q', title="Number of Reviews"),
            color=alt.Color('Sentiment:N', scale=alt.Scale(domain=['POSITIVE', 'NEGATIVE'], range=['#2ca02c', '#d62728'])),
            tooltip=['Sentiment', 'Count', alt.Tooltip('Average Confidence:Q', format='.2%')]
        ).properties(height=400)

        st.altair_chart(chart, use_container_width=True)
