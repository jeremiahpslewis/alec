import os

import altair as alt
import pandas as pd
import streamlit as st

st.title("ALEC: Active Learning Experiment Credit")

simulation_ids_1 = pd.Series(
    [f.split(".")[0] for f in os.listdir("data/applications") if not f.startswith(".")]
)
simulation_ids_2 = pd.Series(
    [f.split(".")[0] for f in os.listdir("data/outcomes") if not f.startswith(".")]
)
simulation_ids_3 = pd.Series(
    [f.split(".")[0] for f in os.listdir("data/portfolios") if not f.startswith(".")]
)

simulation_ids = simulation_ids_1.append(simulation_ids_2).append(simulation_ids_3)

simulation_ids = simulation_ids.value_counts().reset_index()
simulation_ids = simulation_ids.loc[simulation_ids.loc[:, 0] == 3, "index"].tolist()

df_summary_full = pd.DataFrame()

for simulation_id in simulation_ids:

    app_df = pd.read_parquet(f"data/applications/{simulation_id}.parquet")
    outcome_df = pd.read_parquet(f"data/outcomes/{simulation_id}.parquet")
    portfolio_df = pd.read_parquet(f"data/portfolios/{simulation_id}.parquet")

    df = pd.merge(
        pd.merge(app_df, portfolio_df, on="application_id", how="left"),
        outcome_df,
        on=["application_id"],
        how="left",
    )

    df.portfolio = df.portfolio.fillna("rejected_application")

    df_summary = df.groupby(
        ["simulation_id", "application_date", "portfolio"]
    ).counterfactual_default.mean()
    df_summary = df_summary.reset_index()

    df_summary_full = df_summary_full.append(df_summary)

c = (
    alt.Chart(df_summary_full)
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
