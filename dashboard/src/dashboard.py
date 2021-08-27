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

s3 = boto3.resource("s3")
s3_alec = s3.Bucket("alec")

if False:
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
                "default",
                title="Default Rate",
                scale=pct_scale,
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
        )
    )

    p1 = p12 + p11 + p1
    p1 = p1.properties(height=500, width=1000, title="Simulation Defaults over Time")

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
    df_long.loc[:, "variable"] = df_long.variable.str.title().str.replace("_", " ")

    # Based on Altair Example: https://altair-viz.github.io/user_guide/transform/density.html#density-transform
    p2 = (
        alt.Chart(
            df_long[
                df_long.variable.isin(
                    ["Idiosyncratic Individual Risk", "Total Default Risk Log Odds"]
                )
            ],
            width=300,
            height=300,
        )
        .transform_density(
            "value",
            groupby=["variable", "application_date"],
            as_=["risk_score", "density"],
            # extent=[0, 1],
        )
        .mark_line(opacity=0.8)
        .encode(
            x=alt.X("risk_score:Q", title="Log Odds Scale"),
            y=alt.Y("density:Q", title="Density"),
            color=alt.Color("application_date:N", title="Application Date"),
        )
        .facet(
            column=alt.Column("variable:N", title="Distribution of Risk Parameters"),
        )
        .resolve_scale(x="independent")
    )
    # p2 = p2.properties(height=500, width=1000)

    st.write(p2)

    p3 = (
        alt.Chart(df, title="Risk Score Distribution")
        .transform_density(
            "income_based_risk",
            groupby=["application_date"],
            as_=["income_based_risk", "density"],
            # extent=[0, 1],
        )
        .mark_line()
        .encode(
            x=alt.X("income_based_risk:Q", title="Income Based Risk (Log Odds Scale)"),
            y=alt.Y("density:Q", title="Density"),
        )
        .facet(
            column=alt.Column("application_date:N", title="Application Date"),
        )
    )

    st.write(p3)

    p4 = (
        alt.Chart(df, title="Risk Score Distribution")
        .transform_density(
            "asset_based_risk",
            groupby=["application_date"],
            as_=["asset_based_risk", "density"],
            # extent=[0, 1],
        )
        .mark_line()
        .encode(
            x=alt.X("asset_based_risk:Q", title="Asset Based Risk (Log Odds Scale)"),
            y=alt.Y("density:Q", title="Density"),
        )
        .facet(
            column=alt.Column("application_date:N", title="Application Date"),
        )
    )

    st.write(p4)

    p5 = (
        alt.Chart(df)
        .mark_point()
        .encode(
            y=alt.Y(
                "total_default_risk",
                title="Default Probability",
                axis=alt.Axis(format="%"),
            ),
            x=alt.X(
                "income_based_risk",
                title="Income Based Risk (Log Odds)",
            ),
        )
        .facet(
            column=alt.Column("application_date:N", title="Application Date"),
        )
    )

    st.write(p5)

    p6 = (
        alt.Chart(df)
        .mark_point()
        .encode(
            y=alt.Y(
                "total_default_risk",
                title="Default Probability",
                axis=alt.Axis(format="%"),
            ),
            x=alt.X(
                "asset_based_risk",
                title="Asset Based Risk (Log Odds)",
            ),
        )
        .facet(
            column=alt.Column("application_date:N", title="Application Date"),
        )
    )

    st.write(p6)


df_summary_full = pd.read_parquet("s3://alec/dashboard/summary_data.parquet")

df_plot = df_summary_full[df_summary_full.application_date > 2020].copy()

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

b = a.properties(height=300, width=250)

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

c = c.properties(height=300, width=250)
c = c.facet(
    row="active_learning_spec:N",
    column="business_to_research_ratio:N",
)
st.write(c)


df_summary_full.loc[df_summary_full.portfolio.isin(["business", "research"])].groupby(
    ["active_learning_spec", "business_to_research_ratio"]
).counterfactual_default.mean()
