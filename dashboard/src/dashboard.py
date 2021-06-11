import os

import altair as alt
import boto3
import pandas as pd
import streamlit as st

st.title("ALEC: Active Learning Experiment Credit")
alt.data_transformers.disable_max_rows()

s3 = boto3.resource("s3")
s3_alec = s3.Bucket("alec")

# Empty bucket of alec objects

simulation_ids = []
scenario_ids = []

if False:
    for i in ["applications", "outcomes", "portfolios", "scenarios"]:
        file_paths_tmp = [f.key for f in s3_alec.objects.filter(Prefix=f"{i}/")]
        simulation_ids_tmp = [
            simulation_id.split("/")[2].split(".")[0] for simulation_id in file_paths_tmp
        ]
        simulation_ids.extend(simulation_ids_tmp)

        scenario_ids_tmp = [simulation_id.split("/")[1] for simulation_id in file_paths_tmp]
        scenario_ids.extend(scenario_ids_tmp)

    simulation_ids = pd.DataFrame({"simulation_id": simulation_ids})
    simulation_ids = simulation_ids.value_counts().reset_index()
    simulation_ids = simulation_ids.loc[
        simulation_ids.loc[:, 0] == simulation_ids.loc[:, 0].max(), "simulation_id"
    ].tolist()

    scenario_ids = pd.DataFrame({"scenario_id": scenario_ids})
    scenario_ids = scenario_ids.value_counts().reset_index()
    scenario_ids = scenario_ids.loc[
        scenario_ids.loc[:, 0] == scenario_ids.loc[:, 0].max(), "scenario_id"
    ].tolist()

    df_summary_full = pd.DataFrame()

    for scenario_id in scenario_ids:#[1:5]:
        for simulation_id in simulation_ids:#[1:5]:

            app_df = pd.read_parquet(
                f"s3://alec/applications/{scenario_id}/{simulation_id}.parquet"
            )
            outcome_df = pd.read_parquet(
                f"s3://alec/outcomes/{scenario_id}/{simulation_id}.parquet"
            )
            portfolio_df = pd.read_parquet(
                f"s3://alec/portfolios/{scenario_id}/{simulation_id}.parquet"
            )
            scenario_df = pd.read_parquet(
                f"s3://alec/scenarios/{scenario_id}/{simulation_id}.parquet"
            )

            df = pd.merge(
                app_df, portfolio_df, on=["application_id", "simulation_id"], how="left"
            )
            df = pd.merge(
                df,
                outcome_df,
                on=["application_id", "simulation_id"],
                how="left",
            )
            df = pd.merge(
                df,
                scenario_df,
                left_on="scenario_id",
                right_on="id",
                how="left",
            )

            groupby_indices = [
                "simulation_id",
                "application_date",
                "application_acceptance_rate",
                "business_to_research_ratio",
                "active_learning_spec",
            ]

            df.portfolio = df.portfolio.fillna("rejected_application")

            df_summary = df.groupby(
                [*groupby_indices, "portfolio"]
            ).counterfactual_default.mean()
            df_summary = df_summary.reset_index()

            df_summary_all = df.groupby(groupby_indices).counterfactual_default.mean()
            df_summary_all = df_summary_all.reset_index()
            df_summary_all["portfolio"] = "full_dataset"

            df_summary_full = df_summary_full.append(df_summary).append(df_summary_all)
            df_summary_full.to_parquet("s3://alec/dashboard/summary_data.parquet")


df_summary_full = pd.read_parquet("s3://alec/dashboard/summary_data.parquet")
df_plot = df_summary_full[df_summary_full.application_date > 2020].copy()
pct_scale = alt.Scale(domain=(0, 0.3))

c = (
    alt.Chart(df_plot)
    .mark_point(opacity=0.1)
    .encode(
        x=alt.X(
            "application_date:N", title="Application Date", axis=alt.Axis(labelAngle=0)
        ),
        y=alt.Y(
            "counterfactual_default",
            title="Default Rate",
            axis=alt.Axis(format="%"),
            scale=pct_scale,
        ),
        color="portfolio",
    )
)


d = (
    alt.Chart(df_plot)
    .mark_line()
    .encode(
        x=alt.X(
            "application_date:N", title="Application Date", axis=alt.Axis(labelAngle=0)
        ),
        y=alt.Y(
            "mean(counterfactual_default)",
            title="Default Rate",
            axis=alt.Axis(format="%"),
            scale=pct_scale,
        ),
        color="portfolio",
    )
)

c = c.mark_errorband(extent="ci", opacity=0.2) + d + d.mark_point()

c = c.properties(height=500, width=1000)
c.facet(
    row="active_learning_spec:N",
    column='business_to_research_ratio:N',
)
st.write(c)
