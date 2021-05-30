import altair as alt
import pandas as pd
import streamlit as st

st.title("ALEC: Active Learning Experiment Credit")

app_df = pd.read_parquet("data/application_df.parquet")
outcome_df = pd.read_parquet("data/outcome_df.parquet")
portfolio_df = pd.read_parquet("data/portfolio_df.parquet")

df = pd.merge(
    pd.merge(app_df, portfolio_df, on="application_id", how="left"),
    outcome_df,
    on="application_id",
    how="left",
)

# NOTE: Remove this once application date is fixed
df.application_date = df.application_date + 2020

df.portfolio = df.portfolio.fillna("rejected_application")

df_summary = df.groupby(["application_date", "portfolio"]).counterfactual_default.mean()
df_summary = df_summary.reset_index()

c = (
    alt.Chart(df_summary)
    .mark_line()
    .encode(
        x=alt.X(
            "application_date:N", title="Application Date", axis=alt.Axis(labelAngle=0)
        ),
        y=alt.Y(
            "counterfactual_default",
            title="Default Rate",
            axis=alt.Axis(format="%"),
            scale=alt.Scale(domain=(0, 0.5)),
        ),
        color="portfolio",
    )
)
c = c + c.mark_point()

c = c.properties(height=500, width=1000)

st.write(c)
