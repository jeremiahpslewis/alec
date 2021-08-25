import altair as alt
import boto3
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="ALEC: Active Learning Experiment Credit",
    page_icon="--",
    layout="wide",
)

st.title("ALEC: Active Learning Experiment Credit")

alt.data_transformers.disable_max_rows()

s3 = boto3.resource("s3")
s3_alec = s3.Bucket("alec")

# Visualize Synthetic Data

simulation_ids = [f.key for f in s3_alec.objects.filter(Prefix="synthetic_data/")]
simulation_ids = [
    simulation_id.split("/")[1].split(".")[0] for simulation_id in simulation_ids
]

df = pd.DataFrame()
for simulation_id in simulation_ids:
    raw_df = pd.read_parquet(f"s3://alec/synthetic_data/{simulation_id}.parquet")
    raw_df = raw_df.loc[raw_df.simulation_id == simulation_id].copy()
    raw_df.reset_index(inplace=True, drop=True)
    df = df.append(raw_df)

df_summary = (
    df.groupby(["application_date", "simulation_id"]).default.mean().reset_index()
)
# Empty bucket of alec objects

p1 = (
    alt.Chart(df_summary)
    .mark_point()
    .encode(
        y=alt.Y(
            "default",  # title="Asset-based Risk", axis=alt.Axis(labelAngle=0)
        ),
        x=alt.X(
            "application_date:N",
            # title="Counterfactual Default",
            # axis=alt.Axis(format="%"),
            # scale=pct_scale,
        ),
        # color="portfolio",
        # color="application_date"
    )
)

p1 = p1.mark_errorband(extent="ci", opacity=0.2) + p1
p1 = p1.properties(height=500, width=1000)

st.write(p1)

df_long = pd.melt(df, id_vars=["application_date", "simulation_id"])
df_long = df_long.loc[
    df_long.variable.isin(
        [
            "income_based_risk",
            "asset_based_risk",
            "idiosyncratic_individual_risk",
            "total_default_risk_log_odds",
        ]
    )
].copy()
df_long.loc[:, "value"] = df_long.value.astype(float)

# Based on Altair Example: https://altair-viz.github.io/user_guide/transform/density.html#density-transform
p2 = (
    alt.Chart(df_long, width=300, height=300)
    .transform_filter("isValid(datum.variable)")
    .transform_density(
        "value",
        groupby=["variable", "application_date"],
        as_=["risk_score", "density"],
        # extent=[0, 1],
    )
    .mark_area(opacity=0.2)
    .encode(x="risk_score:Q", y="density:Q", color="application_date:N")
    .facet(
        column="variable:N",
    )
)
# p2 = p2.properties(height=500, width=1000)

st.write(p2)


p3 = (
    alt.Chart(df)
    .mark_point()
    .encode(
        y=alt.Y(
            "total_default_risk",  # title="Asset-based Risk", axis=alt.Axis(labelAngle=0)
        ),
        x=alt.X(
            "income_based_risk",
            # title="Counterfactual Default",
            # axis=alt.Axis(format="%"),
            # scale=pct_scale,
        ),
        # color="portfolio",
        # color="application_date"
    )
    .facet(
        column="application_date:N",
    )
)

st.write(p3)

p4 = (
    alt.Chart(df)
    .mark_point()
    .encode(
        y=alt.Y(
            "total_default_risk",  # title="Asset-based Risk", axis=alt.Axis(labelAngle=0)
        ),
        x=alt.X(
            "asset_based_risk",
            # title="Counterfactual Default",
            # axis=alt.Axis(format="%"),
            # scale=pct_scale,
        ),
        # color="portfolio",
        # color="application_date"
    )
    .facet(
        column="application_date:N",
    )
)

st.write(p4)

simulation_ids = []
scenario_ids = []

if True:
    for i in ["applications", "outcomes", "portfolios", "scenarios"]:
        file_paths_tmp = [f.key for f in s3_alec.objects.filter(Prefix=f"{i}/")]
        simulation_ids_tmp = [
            simulation_id.split("/")[2].split(".")[0]
            for simulation_id in file_paths_tmp
        ]
        simulation_ids.extend(simulation_ids_tmp)

        scenario_ids_tmp = [
            simulation_id.split("/")[1] for simulation_id in file_paths_tmp
        ]
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

    for scenario_id in scenario_ids:
        for simulation_id in simulation_ids:
            print(f"Scenario: {scenario_id}, Simulation: {simulation_id}")
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

# c = (
#     alt.Chart(df)
#     .mark_point()
#     .encode(
#         y=alt.Y(
#             "income_based_risk", title="Asset-based Risk", axis=alt.Axis(labelAngle=0)
#         ),
#         x=alt.X(
#             "total_default_risk",
#             title="Counterfactual Default",
#             # axis=alt.Axis(format="%"),
#             # scale=pct_scale,
#         ),
#         # color="portfolio",
#         color="application_date"
#     )
# )

# st.write(c)


df_plot = df_summary_full[df_summary_full.application_date > 2020].copy()
pct_scale = alt.Scale(domain=(0, 1))

a = (
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

b = a.properties(height=500, width=1000)

b = b.facet(
    column="portfolio:N",
)

st.write(b)

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

c = a.mark_errorband(extent="ci", opacity=0.2) + d + d.mark_point()

c = c.properties(height=500, width=1000)
c = c.facet(
    row="active_learning_spec:N",
    column="business_to_research_ratio:N",
)
st.write(c)
