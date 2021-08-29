import os

import altair as alt
import boto3
import pandas as pd
import streamlit as st

pct_scale = alt.Scale(domain=(0, 1))

st.set_page_config(
    page_title="ALEC: Active Learning Experiment Credit",
    page_icon="--",
    layout="wide",
)

st.title("ALEC: Active Learning Experiment Credit")

alt.data_transformers.disable_max_rows()

bucket_name = os.getenv("S3_BUCKET_NAME")

s3 = boto3.resource("s3")
s3_alec = s3.Bucket(bucket_name)

# Visualize Synthetic Data

simulation_ids = [f.key for f in s3_alec.objects.filter(Prefix="synthetic_data/")]
simulation_ids = [
    simulation_id.split("/")[1].split(".")[0] for simulation_id in simulation_ids
]

df = pd.DataFrame()

for simulation_id in simulation_ids:
    raw_df = pd.read_parquet(
        f"s3://{bucket_name}/synthetic_data/{simulation_id}.parquet"
    )
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
            "default",
            title="Default Rate",
            scale=pct_scale,
            axis=alt.Axis(format="%"),
        ),
        x=alt.X(
            "application_date:N",
            title="Application Date",
            axis=alt.Axis(labelAngle=0),
        ),
        color=alt.Color("simulation_id", title="Simulation"),
    )
)

p11 = (
    alt.Chart(df_summary)
    .mark_errorband(extent="ci", opacity=0.2)
    .encode(
        y=alt.Y(
            "default",
            title="Default Rate",
            scale=pct_scale,
            axis=alt.Axis(format="%"),
        ),
        x=alt.X(
            "application_date:N",
            title="Application Date",
            axis=alt.Axis(labelAngle=0),
        ),
    )
)

p12 = p11.mark_line().encode(
    y=alt.Y(
        "mean(default)",
        title="Default Rate",
        scale=pct_scale,
        axis=alt.Axis(format="%"),
    )
)

p1 = p12 + p11 + p1
p1 = p1.properties(height=500, width=1000, title="Simulation Defaults over Time")

st.write(p1)

df_long = pd.melt(df, id_vars=["application_date", "simulation_id"])
df_long = df_long.loc[
    df_long.variable.isin(
        [
            "age",
            "idiosyncratic_individual_risk",
            "total_default_risk_log_odds",
        ]
    )
].copy()
df_long.loc[:, "value"] = df_long.value.astype(float)
df_long.loc[:, "variable"] = df_long.variable.str.title().str.replace("_", " ")

# Based on Altair Example: https://altair-viz.github.io/user_guide/transform/density.html#density-transform
p2 = (
    alt.Chart(
        df_long[
            df_long.variable.isin(
                [
                    "Idiosyncratic Individual Risk",
                    "Total Default Risk Log Odds",
                    "Age",
                ]
            )
        ],
        width=100,
        height=100,
    )
    .transform_density(
        "value",
        groupby=["variable", "application_date"],
        as_=["risk_score", "density"],
    )
    .mark_line(opacity=0.8)
    .encode(
        x=alt.X("risk_score:Q", title="Log Odds Scale"),
        y=alt.Y("density:Q", title="Density"),
        color=alt.Color("variable:N", title="Application Date"),
    )
    .facet(
        column=alt.Column("application_date:N", title="Application Date"),
        row=alt.Row("variable", title="Risk Parameter"),
    )
    .resolve_scale(x="independent", y="independent")
)

st.write(p2)

p3 = (
    alt.Chart(df, height=200, width=200, title="Risk Score Distribution")
    .transform_density(
        "age",
        groupby=["application_date"],
        as_=["total_default_risk", "density"],
        extent=[0, 1],
    )
    .mark_line()
    .encode(
        x=alt.X(
            "total_default_risk:Q",
            title="Default Probability",
            axis=alt.Axis(format="%"),
        ),
        y=alt.Y("density:Q", title="Density"),
    )
    .facet(
        column=alt.Column("application_date:N", title="Application Date"),
    )
    .resolve_scale(y="independent")
)

st.write(p3)

p5 = (
    alt.Chart(df, height=200, width=200)
    .mark_point()
    .encode(
        y=alt.Y(
            "total_default_risk",
            title="Default Probability",
            axis=alt.Axis(format="%"),
        ),
        x=alt.X(
            "age",
            title="Age Based Risk (Log Odds)",
        ),
    )
    .facet(
        column=alt.Column("application_date:N", title="Application Date"),
    )
)

st.write(p5)

df_summary_full = pd.read_parquet(f"s3://{bucket_name}/dashboard/summary_data.parquet")

df_plot = df_summary_full[df_summary_full.application_date > 2020].copy()

pp_scale = alt.Scale(domain=(-1, 1))


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
        color=alt.Color("portfolio", title="Portfolio"),
    )
)

b = a.properties(height=300, width=250)

b = b.facet(
    column=alt.Column("portfolio:N", title="Portfolio"),
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
        color=alt.Color("portfolio", title="Portfolio"),
    )
)

c = a.mark_errorband(extent="ci", opacity=0.2) + d + d.mark_point()

c = c.properties(height=300, width=250)
c = c.facet(
    row=alt.Row("active_learning_spec:N", title="Active Learning Spec"),
    column=alt.Column("research_acceptance_rate:N", title="Research Acceptance Rate"),
)
st.write(c)


no_active_learning_results = (
    df_summary_full.loc[
        df_summary_full.portfolio.isin(["business", "research"])
        & (df_summary_full.research_acceptance_rate == 0)
    ]
    .groupby("simulation_id")
    .counterfactual_default.mean()
).reset_index()

no_active_learning_results.rename(
    columns={"counterfactual_default": "no_active_learning_default"}, inplace=True
)
# Business Portfolio Uplift

active_learning_results = (
    df_summary_full.loc[df_summary_full.portfolio.isin(["business", "research"])]
    .groupby(
        [
            "simulation_id",
            "active_learning_spec",
            "research_acceptance_rate",
            "portfolio",
        ]
    )
    .counterfactual_default.mean()
    .reset_index()
)

active_learning_results = pd.merge(
    active_learning_results, no_active_learning_results, on="simulation_id", how="left"
)

# Net Uplift with Research Portfolio
active_learning_results["net_default_rate_effect"] = (
    active_learning_results["counterfactual_default"]
    - active_learning_results["no_active_learning_default"]
)

a = (
    alt.Chart(
        active_learning_results.loc[active_learning_results.portfolio == "business"]
    )
    .mark_point(opacity=0.1)
    .encode(
        x=alt.X(
            "research_acceptance_rate:N",
            title="Business to Research Ratio",
            axis=alt.Axis(labelAngle=0),
        ),
        y=alt.Y(
            "net_default_rate_effect",
            title="Net Default Rate Effect (Business Portfolio), in p.p.",
            axis=alt.Axis(format="%"),
        ),
        color="portfolio",
    )
)

d = a.encode(
    y=alt.Y(
        "mean(net_default_rate_effect)",
        title="Net Default Rate Effect (Business Portfolio), in p.p.",
    )
)


b = d.mark_errorband(extent="ci", opacity=0.2) + d.mark_line() + a

b = b.properties(height=300, width=250)

b = b.facet(
    column=alt.Column("active_learning_spec:N", title="Active Learning Spec"),
)

st.write(b)
