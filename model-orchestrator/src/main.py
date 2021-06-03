import os

import pandas as pd
from dagster import ModeDefinition, PresetDefinition, execute_pipeline, pipeline, solid
from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

# NOTE: counterfactual_default is defined as default outcome had applicant been granted loan
run_metadata = ["simulation_id"]
simulation_metadata = ["application_date", "application_id", "counterfactual_default"]
X_vars = ["individual_default_risk", "business_cycle_default_risk"]
y_var = ["default"]

full_application_col_set = [*run_metadata, *simulation_metadata, *X_vars]
full_portfolio_col_set = [
    *run_metadata,
    "application_id",
    "portfolio",
    "credit_granted",
    "funding_probability",
]
full_outcome_col_set = [*run_metadata, "application_id", "default"]


def get_raw_data(simulation_id):
    raw_df = pd.read_parquet(
        f"data/synthetic-data/synthetic_data_{simulation_id}.parquet"
    )
    raw_df = raw_df.loc[raw_df.simulation_id == simulation_id].copy()
    raw_df.reset_index(inplace=True, drop=True)
    raw_df["counterfactual_default"] = raw_df["default"]
    return raw_df


def get_historical_data(
    simulation_id,
) -> list[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = get_raw_data(simulation_id)
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


@solid(config_schema={"simulation_id": str})
def get_historical_application_data(context):
    simulation_id = context.solid_config["simulation_id"]

    return get_historical_data(simulation_id)["applications"]


@solid(config_schema={"simulation_id": str})
def get_historical_portfolio_data(context):
    simulation_id = context.solid_config["simulation_id"]

    return get_historical_data(simulation_id)["portfolio"]


@solid(config_schema={"simulation_id": str})
def get_historical_outcome_data(context):
    simulation_id = context.solid_config["simulation_id"]

    return get_historical_data(simulation_id)["outcomes"]


@solid
def get_model_pipeline(context, model_spec: int = 1) -> Pipeline:
    if model_spec == 1:
        column_trans = ColumnTransformer(
            [
                (
                    "individual_default_risk",
                    StandardScaler(),
                    ["individual_default_risk"],
                ),
                (
                    "business_cycle_default_risk",
                    StandardScaler(),
                    ["business_cycle_default_risk"],
                ),
            ],
            remainder="drop",
        )
        model_pipeline = make_pipeline(column_trans, linear_model.LogisticRegression())
    else:
        pass
    return model_pipeline


@solid
def get_active_learning_pipeline(context, active_learning_spec: int = 1) -> Pipeline:
    if active_learning_spec == 1:
        column_trans = ColumnTransformer(
            [
                (
                    "individual_default_risk",
                    StandardScaler(),
                    ["individual_default_risk"],
                ),
                (
                    "business_cycle_default_risk",
                    StandardScaler(),
                    ["business_cycle_default_risk"],
                ),
            ],
            remainder="drop",
        )
        model_pipeline = make_pipeline(column_trans, linear_model.LogisticRegression())
    else:
        pass
    return model_pipeline


@solid
def train_model(
    context,
    application_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    outcome_df,
    model_pipeline: Pipeline,
) -> Pipeline:
    """
    training_data: data collected from previous loans granted, as pd.DataFrame
    model: machine learning model (pipeline) which can be applied to training data
    """
    training_df = pd.merge(
        application_df, portfolio_df, on=["application_id", "simulation_id"], how="left"
    )

    training_df = pd.merge(
        training_df,
        outcome_df,
        on=["application_id", "simulation_id"],
        how="left",
    )

    # NOTE: Check here that rows are not dropped / added!
    training_df.to_parquet("training_df.parquet")
    application_df.to_parquet("application_df.parquet")
    portfolio_df.to_parquet("portfolio_df.parquet")
    outcome_df.to_parquet("outcome_df.parquet")

    assert (
        training_df.application_id.duplicated().sum() == 0
    ), training_df.application_date.max()
    assert (
        training_df.shape[0] == application_df.shape[0]
    ), training_df.simulation_id.value_counts()
    assert training_df.portfolio.notnull().sum() == portfolio_df.shape[0]
    assert training_df.default.notnull().sum() == outcome_df.shape[0]
    assert training_df.shape[0] > 0

    # NOTE: Currently all cases without observed default are dropped for ML model!

    training_df = training_df.loc[training_df.default.notnull()].copy()

    # try:
    model_pipeline.fit(training_df.loc[:, X_vars], training_df["default"].astype("int"))
    # except Exception:
    #     training_df.to_parquet(f"debug_{training_df.application_date.max()}.parquet")

    return model_pipeline


@solid(config_schema={"application_date": int, "simulation_id": str})
def get_applications(context, application_df) -> pd.DataFrame:
    """
    gets applications for new loans from customers
    """

    application_date = context.solid_config["application_date"]
    simulation_id = context.solid_config["simulation_id"]

    raw_application_df = get_raw_data(simulation_id)
    new_application_df = raw_application_df.loc[
        raw_application_df.application_date == application_date
    ].copy()
    new_application_df.reset_index(inplace=True, drop=True)
    return application_df.append(
        new_application_df[full_application_col_set]
    ).reset_index(drop=True)


@solid(config_schema={"application_date": int})
def choose_business_portfolio(
    context,
    application_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    model_pipeline: Pipeline,
    application_acceptance_rate: float = 0.1,
) -> pd.DataFrame:
    """
    Decide whom to grant loans to (for profit)

    applications: pd.DataFrame
    model: machine learning model (pipeline) which can be applied to applications, based on training data
    """

    application_date = context.solid_config["application_date"]

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


@solid(config_schema={"application_date": int})
def choose_research_portfolio(
    context,
    application_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
    model_pipeline: Pipeline,
    active_learning_pipeline: Pipeline,
    n_research_budget: int = 20,
) -> pd.DataFrame:
    """
    Decide whom to grant loans to (for research / profit in subsequent rounds)

    business_portfolio: {"application_id", "credit_granted"} as pd.DataFrame
    """

    application_date = context.solid_config["application_date"]

    unfunded_applications = application_df[
        ~application_df.application_id.isin(portfolio_df.application_id.tolist())
        & (application_df.application_date == application_df.application_date.max())
    ]

    # NOTE: No applications this application_date!
    if unfunded_applications.shape[0] == 0:
        return portfolio_df

    research_portfolio_df = unfunded_applications.sample(
        min(n_research_budget, unfunded_applications.shape[0])
    )[["application_id", "simulation_id"]].reset_index(drop=True)

    research_portfolio_df["portfolio"] = "research"
    research_portfolio_df["credit_granted"] = True
    research_portfolio_df["funding_probability"] = (
        n_research_budget / unfunded_applications.shape[0]
    )
    return portfolio_df.append(research_portfolio_df[full_portfolio_col_set])


@solid(config_schema={"application_date": int, "simulation_id": str})
def observe_outcomes(
    context, portfolio_df: pd.DataFrame, outcome_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Observe outcomes to granted credit.
    """
    application_date = context.solid_config["application_date"]
    simulation_id = context.solid_config["simulation_id"]

    raw_data = get_raw_data(simulation_id)
    new_loan_outcomes = raw_data.loc[
        (raw_data.application_date == application_date)
        & (raw_data.application_id.isin(portfolio_df.application_id.tolist()))
    ].copy()
    return outcome_df.append(new_loan_outcomes[full_outcome_col_set])


@solid(config_schema={"simulation_id": str})
def export_results(
    context,
    application_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
):
    simulation_id = context.solid_config["simulation_id"]

    application_df.to_parquet(f"data/applications/{simulation_id}.parquet")
    portfolio_df.to_parquet(f"data/portfolios/{simulation_id}.parquet")
    outcome_df.to_parquet(f"data/outcomes/{simulation_id}.parquet")


def var_if_gr_1(i, var):
    if i > 1:
        return f"{var}_{i}"
    else:
        return var


def run_simulation(simulation_id):
    solids_dict = {
        var_if_gr_1(i + 1, var): {"config": {"application_date": range(2021, 2031)[i]}}
        for i in range(10)
        for var in [
            "choose_business_portfolio",
            "choose_research_portfolio",
        ]
    }

    solids_dict.update(
        {
            var_if_gr_1(i + 1, var): {
                "config": {
                    "application_date": range(2021, 2031)[i],
                    "simulation_id": simulation_id,
                }
            }
            for i in range(10)
            for var in [
                "get_applications",
                "observe_outcomes",
            ]
        }
    )

    solids_dict.update(
        {
            var: {"config": {"simulation_id": simulation_id}}
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

        for t in range(10):

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
    for d in ["data/applications", "data/portfolios", "data/outcomes"]:
        os.mkdir(d)
    simulation_ids = [
        f for f in os.listdir("data/synthetic-data") if not f.startswith(".")
    ]
    simulation_ids = [
        simulation_id.split("_")[2].split(".")[0] for simulation_id in simulation_ids
    ]

    for simulation_id in simulation_ids:
        run_simulation(simulation_id)
