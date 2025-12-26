import streamlit as st
import pandas as pd
import json
from transformers import pipeline
import altair as alt

# 1. Page Configuration
st.set_page_config(page_title="Brand Reputation Monitor 2023", layout="wide")

# 2. Data Loading Function (Optimized for your specific JSON structure)
@st.cache_data
def load_data():
    with open("scraped_data.json", "r") as f:
        raw_data = json.load(f)
    
    # Process Products
    df_prod = pd.DataFrame(raw_data["products"])
    df_prod["type"] = "product"
    
    # Process Testimonials
    df_test = pd.DataFrame(raw_data["testimonials"])
    df_test["type"] = "testimonial"
    
    # Process Reviews (Unpacking the nested edges/nodes)
    reviews_list = []
    try:
        edges = raw_data["reviews"]["data"]["reviews"]["edges"]
        for edge in edges:
            reviews_list.append(edge["node"])
    except KeyError:
        st.error("Could not find the 'reviews' path in your JSON file.")

    df_rev = pd.DataFrame(reviews_list)
    df_rev["type"] = "review"
    
    # Combine everything into one Master Dataframe
    df_final = pd.concat([df_prod, df_test, df_rev], axis=0, ignore_index=True, sort=False)
    
    # Convert date strings to Python datetime objects
    df_final['date'] = pd.to_datetime(df_final['date'])
    
    return df_final

# --- CRITICAL: Load data globally so 'df' is defined for all pages ---
try:
    df = load_data()
except FileNotFoundError:
    st.error("File 'scraped_data.json' not found. Please ensure it is in the same folder.")
    st.stop()

# 3. Sidebar Navigation
st.sidebar.title("Brand Monitor 2023")
page = st.sidebar.radio("Navigate to:", ["Products", "Testimonials", "Reviews"])

# 4. Section Behavior
if page == "Products":
    st.header("📦 Product Catalog")
    products_df = df[df['type'] == 'product'].copy()
    # Clean up display: remove internal columns
    display_cols = ["name", "price", "short-description"]
    st.dataframe(products_df[display_cols], use_container_width=True, hide_index=True)

elif page == "Testimonials":
    st.header("💬 Customer Testimonials")
    testimonials_df = df[df['type'] == 'testimonial'].copy()
    # Only show columns relevant to testimonials
    st.dataframe(testimonials_df[["text", "rating"]].dropna(subset=["text"]), use_container_width=True, hide_index=True)
    #st.table(testimonials_df[["text", "rating"]].dropna(subset=["text"]))

elif page == "Reviews":
    st.header("📊 2023 Review Sentiment Analysis")

    # --- Month Selection Slider ---
    months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]
    
    selected_month_name = st.select_slider(
        "Filter reviews by month (2023):",
        options=months,
        value="May" # Default to May since your data has May reviews
    )

    # Convert month name to number (Jan=1, Dec=12)
    month_num = months.index(selected_month_name) + 1
    
    # Filter the Master Dataframe
    filtered_reviews = df[
        (df['type'] == 'review') & 
        (df['date'].dt.month == month_num) & 
        (df['date'].dt.year == 2023)
    ].copy()

    if filtered_reviews.empty:
        st.warning(f"No reviews found for {selected_month_name} 2023.")
    else:
        st.write(f"Found **{len(filtered_reviews)}** reviews for {selected_month_name}:")
        
        # Display the filtered reviews
        st.dataframe(filtered_reviews[["date", "text", "rating"]], use_container_width=True, hide_index=True)

        # ---------------------------------------------------------
        # PART 3: Sentiment Analysis (Hugging Face)
        # ---------------------------------------------------------
        st.divider()
        st.subheader("Sentiment Analysis Results")

        # 1. Load the model (Cached so it only happens once)
        @st.cache_resource
        def load_sentiment_model():
            # This specific model is fast and accurate for Positive/Negative tasks
            return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

        sentiment_pipeline = load_sentiment_model()

        # 2. Perform Analysis
        # We extract the 'text' column as a list and pass it to the model
        texts = filtered_reviews["text"].tolist()

        with st.spinner('Analyzing sentiment...'):
            results = sentiment_pipeline(texts)

        # Store numeric score for calculations and string for display
        filtered_reviews["sentiment"] = [res["label"] for res in results]
        filtered_reviews["score_val"] = [res["score"] for res in results]
        filtered_reviews["confidence"] = [f"{res['score']:.2%}" for res in results]

        def color_sentiment(val):
            color = '#90ee90' if val == 'POSITIVE' else '#ffcccb'
            return f'background-color: {color}'

        styled_df = filtered_reviews[["date", "text", "sentiment", "confidence"]].style.applymap(
            color_sentiment, subset=['sentiment']
        )
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Metrics
        pos_count = (filtered_reviews["sentiment"] == "POSITIVE").sum()
        neg_count = (filtered_reviews["sentiment"] == "NEGATIVE").sum()
        col1, col2 = st.columns(2)
        col1.metric("Positive Reviews", pos_count)
        col2.metric("Negative Reviews", neg_count)

        # ---------------------------------------------------------
        # PART 4: Visualization (Altair)
        # ---------------------------------------------------------
        st.divider()
        st.subheader(f"Sentiment Distribution: {selected_month_name}")

        # Group data to get Count and Average Confidence
        chart_df = filtered_reviews.groupby("sentiment")["score_val"].agg(['count', 'mean']).reset_index()
        chart_df.columns = ["Sentiment", "Count", "Average Confidence"]

        # Create Altair Bar Chart
        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X('Sentiment:N', title="Sentiment Type"),
            y=alt.Y('Count:Q', title="Number of Reviews"),
            color=alt.Color('Sentiment:N', scale=alt.Scale(domain=['POSITIVE', 'NEGATIVE'], range=['#2ca02c', '#d62728'])),
            tooltip=[
                'Sentiment', 
                'Count', 
                alt.Tooltip('Average Confidence:Q', format='.2%') # This handles the advanced requirement
            ]
        ).properties(width='container', height=400)

        st.altair_chart(chart, use_container_width=True)