import streamlit as st

st.title("My first app!")

import altair as alt
import pandas as pd

source = pd.DataFrame(
    {
        "a": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
        "b": [28, 55, 43, 91, 81, 53, 19, 87, 52],
    }
)

c = alt.Chart(source).mark_bar().encode(x="a", y="b")

st.write(c)

app_df = pd.read_parquet("data/application_df.parquet")
outcome_df = pd.read_parquet("data/outcome_df.parquet")
portfolio_df = pd.read_parquet("data/portfolio_df.parquet")

df = pd.merge(
    pd.merge(app_df, portfolio_df, on="application_id", how="left"),
    outcome_df,
    on="application_id",
    how="left",
)

df.groupby("application_date")
df.portfolio.value_counts()
# df[].groupby("application_date").
