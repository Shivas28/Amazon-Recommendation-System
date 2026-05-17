import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Product Recommendation System", layout="wide")

st.title("🛒 Product Recommendation System")
st.markdown("**Amazon Ratings Dataset — User-Based Collaborative Filtering**")
st.markdown("---")

@st.cache_data
def load_data():
    df = pd.read_csv("ratings_.csv", names=["user_id", "prod_id", "rating", "timestamp"])
    df.drop("timestamp", axis=1, inplace=True)
    user_counts = df["user_id"].value_counts()
    prod_counts  = df["prod_id"].value_counts()
    df = df[df["user_id"].isin(user_counts[user_counts >= 50].index)]
    df = df[df["prod_id"].isin(prod_counts[prod_counts >= 5].index)]
    return df

@st.cache_data
def build_sim(df):
    matrix = df.pivot_table(index="user_id", columns="prod_id", values="rating", fill_value=0)
    sim = cosine_similarity(matrix)
    return pd.DataFrame(sim, index=matrix.index, columns=matrix.index)

df = load_data()
sim_df = build_sim(df)

st.write(f"Dataset: {len(df):,} ratings  |  {df['user_id'].nunique():,} users  |  {df['prod_id'].nunique():,} products")
st.markdown("---")

col1, col2 = st.columns([2, 1])
with col1:
    user_id = st.selectbox("Select a User ID", df["user_id"].unique())
with col2:
    n = st.slider("Number of Recommendations", 1, 10, 5)

if st.button("Get Recommendations", use_container_width=True):
    similar_users = sim_df[user_id].sort_values(ascending=False).iloc[1:6].index
    already_rated = df[df["user_id"] == user_id]["prod_id"].values
    recs = df[df["user_id"].isin(similar_users) & ~df["prod_id"].isin(already_rated)]
    top = recs.groupby("prod_id")["rating"].mean().sort_values(ascending=False).head(n)

    st.subheader(f"Top {n} Recommendations for {user_id}")
    result = top.reset_index()
    result.columns = ["Product ID", "Avg Rating"]
    result.index += 1
    st.dataframe(result, use_container_width=True)

    st.markdown("---")
    st.caption("Recommendations are based on users with similar rating patterns (cosine similarity).")
