# Must be loaded first, else 'free()' error
from dagster import ModeDefinition, PresetDefinition, execute_pipeline, pipeline, solid

import os
from typing import Union

import boto3
import modAL
import numpy as np
import pandas as pd
from modAL.models import ActiveLearner
import sklearn
from modAL.uncertainty import uncertainty_sampling
from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from yaml import safe_load

# NOTE: counterfactual_default is defined as default outcome had applicant been granted loan
simulation_indices = ["simulation_id", "application_id"]
simulation_metadata = [
    "counterfactual_default",
    "scenario_id",
    "idiosyncratic_individual_risk",
    "total_default_risk",
    "income_based_risk_var",
    "asset_based_risk_var",
    "application_date",
]

X_vars = ["income_based_risk", "asset_based_risk"]
y_var = ["default"]

full_application_col_set = [*simulation_indices, *simulation_metadata, *X_vars]
full_portfolio_col_set = [
    *simulation_indices,
    "portfolio",
    "credit_granted",
    "funding_probability",
]
full_outcome_col_set = [*simulation_indices, "default"]


def get_scenario_df():
    with open("scenarios.yml", "r") as f:
        scenarios = safe_load(f)
        scenario_df = pd.DataFrame(scenarios["scenarios"])
    return scenario_df


def get_raw_data(simulation_id, scenario_id):
    raw_df = pd.read_parquet(f"s3://alec/synthetic_data/{simulation_id}.parquet")
    raw_df = raw_df.loc[raw_df.simulation_id == simulation_id].copy()
    raw_df.reset_index(inplace=True, drop=True)
    raw_df["counterfactual_default"] = raw_df["default"]
    raw_df["scenario_id"] = scenario_id
    return raw_df


def get_historical_data(
    simulation_id, scenario_id
) -> list[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = get_raw_data(simulation_id, scenario_id)
    df["portfolio"] = "business"
    df["credit_granted"] = True
    df["funding_probability"] = 1

    df_hist = (
        df.loc[df.application_date == df.application_date.min()]
        .copy()
        .reset_index(drop=True)
    )

    hist_application_df = (
        df_hist.loc[
            :,
            full_application_col_set,
        ]
        .copy()
        .reset_index(drop=True)
    )

    hist_portfolio_df = (
        df_hist.loc[:, full_portfolio_col_set].copy().reset_index(drop=True)
    )

    hist_outcome_df = df_hist.loc[:, full_outcome_col_set].copy().reset_index(drop=True)

    return {
        "applications": hist_application_df,
        "portfolio": hist_portfolio_df,
        "outcomes": hist_outcome_df,
    }


@solid(config_schema={"simulation_id": str, "scenario_id": str})
def get_historical_application_data(context):
    simulation_id = context.solid_config["simulation_id"]
    scenario_id = context.solid_config["scenario_id"]

    return get_historical_data(simulation_id, scenario_id)["applications"]


@solid(config_schema={"simulation_id": str, "scenario_id": str})
def get_historical_portfolio_data(context):
    simulation_id = context.solid_config["simulation_id"]
    scenario_id = context.solid_config["scenario_id"]

    return get_historical_data(simulation_id, scenario_id)["portfolio"]


@solid(config_schema={"simulation_id": str, "scenario_id": str})
def get_historical_outcome_data(context):
    simulation_id = context.solid_config["simulation_id"]
    scenario_id = context.solid_config["scenario_id"]

    return get_historical_data(simulation_id, scenario_id)["outcomes"]


def get_feature_pipeline():
    column_trans = ColumnTransformer(
        [
            (
                "asset_based_risk",
                "passthrough",
                ["asset_based_risk"],
            ),
            (
                "income_based_risk",
                "passthrough",
                ["income_based_risk"],
            ),
        ],
        remainder="drop",
    )
    return column_trans


def get_model_pipeline_object():
    column_trans = get_feature_pipeline()
    model_pipeline = make_pipeline(column_trans, linear_model.LogisticRegression())
    return model_pipeline


@solid(config_schema={"scenario_id": str})
def get_model_pipeline(context) -> sklearn.pipeline.Pipeline:
    scenario_id = context.solid_config["scenario_id"]
    scenario_df = get_scenario_df()

    ml_model_spec = scenario_df.loc[
        scenario_df.id == scenario_id, "ml_model_spec"
    ].iloc[0]
    column_trans = get_feature_pipeline()

    if ml_model_spec == 1:
        model_pipeline = get_model_pipeline_object()
    else:
        model_pipeline = get_model_pipeline_object()
    return model_pipeline


@solid(config_schema={"scenario_id": str})
def get_active_learning_pipeline(context):
    scenario_id = context.solid_config["scenario_id"]
    scenario_df = get_scenario_df()

    active_learning_spec = scenario_df.loc[
        scenario_df.id == scenario_id, "active_learning_spec"
    ].iloc[0]
    model_pipeline = get_model_pipeline_object()

    if active_learning_spec == "random":
        return None
    elif active_learning_spec != "random":
        return getattr(modAL.uncertainty, active_learning_spec)
    return model_pipeline


def prepare_training_data(
    application_df: pd.DataFrame, portfolio_df: pd.DataFrame, outcome_df
):
    training_df = pd.merge(
        application_df, portfolio_df, on=["application_id", "simulation_id"], how="left"
    )

    training_df = pd.merge(
        training_df,
        outcome_df,
        on=["application_id", "simulation_id"],
        how="left",
    )

    assert (
        training_df.application_id.duplicated().sum() == 0
    ), training_df.application_date.max()
    assert (
        training_df.shape[0] == application_df.shape[0]
    ), training_df.simulation_id.value_counts()
    assert training_df.portfolio.notnull().sum() == portfolio_df.shape[0]
    assert training_df.default.notnull().sum() == outcome_df.shape[0]
    assert training_df.shape[0] > 0

    return training_df.reset_index(drop=True)


@solid
def train_model(
    context,
    application_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    outcome_df,
    model_pipeline: sklearn.pipeline.Pipeline,
) -> sklearn.pipeline.Pipeline:
    """
    training_data: data collected from previous loans granted, as pd.DataFrame
    model: machine learning model (pipeline) which can be applied to training data
    """
    training_df = prepare_training_data(application_df, portfolio_df, outcome_df)

    # NOTE: Currently all cases without observed default are dropped for ML model!

    training_df = training_df.loc[training_df.default.notnull()].copy()

    # try:
    model_pipeline.fit(training_df.loc[:, X_vars], training_df["default"].astype("int"))
    # except Exception:
    #     training_df.to_parquet(f"debug_{training_df.application_date.max()}.parquet")

    return model_pipeline


@solid(
    config_schema={"application_date": int, "simulation_id": str, "scenario_id": str}
)
def get_applications(context, application_df) -> pd.DataFrame:
    """
    gets applications for new loans from customers
    """

    application_date = context.solid_config["application_date"]
    simulation_id = context.solid_config["simulation_id"]
    scenario_id = context.solid_config["scenario_id"]

    raw_application_df = get_raw_data(simulation_id, scenario_id)
    new_application_df = raw_application_df.loc[
        raw_application_df.application_date == application_date
    ].copy()
    new_application_df.reset_index(inplace=True, drop=True)
    return application_df.append(
        new_application_df[full_application_col_set]
    ).reset_index(drop=True)


@solid(config_schema={"application_date": int, "scenario_id": str})
def choose_business_portfolio(
    context,
    application_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    model_pipeline: sklearn.pipeline.Pipeline,
) -> pd.DataFrame:
    """
    Decide whom to grant loans to (for profit)

    applications: pd.DataFrame
    model: machine learning model (pipeline) which can be applied to applications, based on training data
    """

    application_date = context.solid_config["application_date"]
    scenario_id = context.solid_config["scenario_id"]
    scenario_df = get_scenario_df()

    application_acceptance_rate = scenario_df.loc[
        scenario_df.id == scenario_id, "application_acceptance_rate"
    ].iloc[0]

    current_application_df = (
        application_df.loc[
            application_df.application_date == application_df.application_date.max()
        ]
        .copy()
        .reset_index(drop=True)
    )

    # NOTE: No applications this application_date!
    if current_application_df.shape[0] == 0:
        return portfolio_df

    current_application_df["est_default_prob"] = pd.DataFrame(
        model_pipeline.predict_proba(current_application_df.loc[:, X_vars])
    ).loc[:, 1]

    assert (
        current_application_df.est_default_prob.isna().sum() == 0
    ), "Some estimated default probabilities NaN"

    business_portfolio_df = (
        current_application_df.loc[
            current_application_df["est_default_prob"].rank(method="first")
            <= int(current_application_df.shape[0] * application_acceptance_rate)
        ]
        .copy()[["application_id", "simulation_id"]]
        .reset_index(drop=True)
    )

    business_portfolio_df["portfolio"] = "business"
    business_portfolio_df["funding_probability"] = 1  # TODO: Change this?
    business_portfolio_df["credit_granted"] = True

    return portfolio_df.append(business_portfolio_df[full_portfolio_col_set])


@solid(config_schema={"application_date": int, "scenario_id": str})
def choose_research_portfolio(
    context,
    application_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
    model_pipeline: sklearn.pipeline.Pipeline,
    active_learning_pipeline,
) -> pd.DataFrame:
    """
    Decide whom to grant loans to (for research / profit in subsequent rounds)

    business_portfolio: {"application_id", "credit_granted"} as pd.DataFrame
    """

    application_date = context.solid_config["application_date"]
    scenario_id = context.solid_config["scenario_id"]
    scenario_df = get_scenario_df()

    business_to_research_ratio = scenario_df.loc[
        scenario_df.id == scenario_id, "business_to_research_ratio"
    ].iloc[0]

    unfunded_applications = application_df[
        ~application_df.application_id.isin(portfolio_df.application_id.tolist())
        & (application_df.application_date == application_df.application_date.max())
    ]

    # NOTE: No applications this application_date!
    if unfunded_applications.shape[0] == 0:
        return portfolio_df

    # NOTE: If business_to_research_ratio == 0, no research loans are made
    if business_to_research_ratio == 0:
        return portfolio_df

    business_loan_count = sum(
        application_df.application_id.isin(portfolio_df.application_id.tolist())
        & (application_df.application_date == application_df.application_date.max())
    )
    n_research_loans = int(business_loan_count * (1 / business_to_research_ratio))

    if active_learning_pipeline is None:
        research_portfolio_df = unfunded_applications.sample(
            min(n_research_loans, unfunded_applications.shape[0])
        )
    else:

        active_learning_df = prepare_training_data(
            unfunded_applications,
            portfolio_df.loc[
                portfolio_df.application_id.isin(
                    unfunded_applications.application_id.tolist()
                )
            ],
            outcome_df.loc[
                outcome_df.application_id.isin(
                    unfunded_applications.application_id.tolist()
                )
            ],
        )

        research_loan_index = active_learning_pipeline(
            classifier=model_pipeline,
            X=active_learning_df.loc[:, X_vars],
            n_instances=n_research_loans,
        )
        research_portfolio_df = active_learning_df.loc[research_loan_index]

    research_portfolio_df = (
        research_portfolio_df[["application_id", "simulation_id"]]
        .reset_index(drop=True)
        .copy()
    )

    research_portfolio_df["portfolio"] = "research"
    research_portfolio_df["credit_granted"] = True
    research_portfolio_df["funding_probability"] = np.nan

    return portfolio_df.append(research_portfolio_df[full_portfolio_col_set])


@solid(
    config_schema={"application_date": int, "simulation_id": str, "scenario_id": str}
)
def observe_outcomes(
    context, portfolio_df: pd.DataFrame, outcome_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Observe outcomes to granted credit.
    """
    application_date = context.solid_config["application_date"]
    simulation_id = context.solid_config["simulation_id"]
    scenario_id = context.solid_config["scenario_id"]

    raw_data = get_raw_data(simulation_id, scenario_id)
    new_loan_outcomes = raw_data.loc[
        (raw_data.application_date == application_date)
        & (raw_data.application_id.isin(portfolio_df.application_id.tolist()))
    ].copy()
    return outcome_df.append(new_loan_outcomes[full_outcome_col_set])


@solid(config_schema={"simulation_id": str, "scenario_id": str})
def export_results(
    context,
    application_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
):
    simulation_id = context.solid_config["simulation_id"]
    scenario_id = context.solid_config["scenario_id"]

    application_df.to_parquet(
        f"s3://alec/applications/{scenario_id}/{simulation_id}.parquet"
    )
    portfolio_df.to_parquet(
        f"s3://alec/portfolios/{scenario_id}/{simulation_id}.parquet"
    )
    outcome_df.to_parquet(f"s3://alec/outcomes/{scenario_id}/{simulation_id}.parquet")
    get_scenario_df().to_parquet(
        f"s3://alec/scenarios/{scenario_id}/{simulation_id}.parquet"
    )


def var_if_gr_1(i, var):
    if i > 1:
        return f"{var}_{i}"
    else:
        return var


def run_simulation(simulation_id, scenario_id):
    solids_dict = {
        var_if_gr_1(i + 1, var): {
            "config": {
                "application_date": range(2021, 2031)[i],
                "scenario_id": scenario_id,
            }
        }
        for i in range(5)
        for var in [
            "choose_business_portfolio",
            "choose_research_portfolio",
        ]
    }

    solids_dict.update(
        {
            "get_model_pipeline": {"config": {"scenario_id": scenario_id}},
            "get_active_learning_pipeline": {"config": {"scenario_id": scenario_id}},
        }
    )

    solids_dict.update(
        {
            var_if_gr_1(i + 1, var): {
                "config": {
                    "application_date": range(2021, 2031)[i],
                    "simulation_id": simulation_id,
                    "scenario_id": scenario_id,
                }
            }
            for i in range(5)
            for var in [
                "get_applications",
                "observe_outcomes",
            ]
        }
    )

    solids_dict.update(
        {
            var: {
                "config": {"simulation_id": simulation_id, "scenario_id": scenario_id}
            }
            for var in [
                "get_historical_application_data",
                "get_historical_portfolio_data",
                "get_historical_outcome_data",
                "export_results",
            ]
        }
    )

    run_config = {"solids": solids_dict}

    @pipeline(
        mode_defs=[ModeDefinition("unittest")],
        preset_defs=[
            PresetDefinition(
                "unittest",
                run_config=run_config,
                mode="unittest",
            )
        ],
    )
    def active_learning_experiment_credit():
        application_df = get_historical_application_data()
        portfolio_df = get_historical_portfolio_data()
        outcome_df = get_historical_outcome_data()
        model_pipeline = get_model_pipeline()
        active_learning_pipeline = get_active_learning_pipeline()

        for t in range(5):

            trained_model = train_model(
                application_df,
                portfolio_df,
                outcome_df,
                model_pipeline,
            )

            application_df = get_applications(application_df)

            portfolio_df = choose_business_portfolio(
                application_df, portfolio_df, trained_model
            )

            portfolio_df = choose_research_portfolio(
                application_df,
                portfolio_df,
                outcome_df,
                trained_model,
                active_learning_pipeline,
            )

            outcome_df = observe_outcomes(portfolio_df, outcome_df)

        export_results(application_df, portfolio_df, outcome_df)

    execute_pipeline(active_learning_experiment_credit, run_config=run_config)


if __name__ == "__main__":
    s3 = boto3.resource("s3")
    s3_alec = s3.Bucket("alec")

    # Empty bucket of alec objects
    for folder in ["applications", "portfolios", "outcomes", "scenarios"]:
        s3_alec.objects.filter(Prefix=f"{folder}/").delete()
        s3_alec.objects.filter(Prefix=f"{folder}/").delete()

    simulation_ids = [f.key for f in s3_alec.objects.filter(Prefix="synthetic_data/")]
    simulation_ids = [
        simulation_id.split("/")[1].split(".")[0] for simulation_id in simulation_ids
    ]

    scenario_df = get_scenario_df()
    for scenario_id in scenario_df.id.tolist():
        for simulation_id in simulation_ids:
            run_simulation(simulation_id, scenario_id)
