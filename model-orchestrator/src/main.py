import pandas as pd
from dagster import pipeline, solid
from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

metadata = ["application_date", "application_id"]
X_vars = ["individual_default_risk", "business_cycle_default_risk"]
y_var = ["default"]

full_application_col_set = [*metadata, *X_vars, *y_var]
full_portfolio_col_set = [
    "application_id",
    "default",
    "portfolio",
    "credit_granted",
    "funding_probability",
]
full_outcome_col_set = ["application_id", "default"]


def get_raw_data():
    raw_df = pd.read_parquet("portfolio_df.parquet")
    return raw_df


@solid
def get_historical_data(
    context, epoch_start: int = 0
) -> list[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = get_raw_data()
    df["portfolio"] = "business"
    df["credit_granted"] = True
    df["funding_probability"] = 1

    hist_application_df = df.loc[
        df.application_date < (epoch_start + 1) * 100,
        full_application_col_set,
    ].copy()

    hist_portfolio_df = df.loc[
        df.application_date < (epoch_start + 1) * 100, full_portfolio_col_set
    ].copy()

    hist_outcome_df = df.loc[
        df.application_date < (epoch_start + 1) * 100, full_portfolio_col_set
    ].copy()

    return {
        "applications": hist_application_df,
        "portfolio": hist_portfolio_df,
        "outcomes": hist_outcome_df,
    }


@solid
def get_model_pipeline(context, model_spec=1) -> Pipeline:
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
def get_active_learning_pipeline(context, active_learning_spec=1) -> Pipeline:
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
        pd.merge(application_df, portfolio_df), outcome_df, on="application_id"
    )

    # NOTE: Check here that rows are not dropped / added!
    assert training_df.shape[0] == application_df.shape[0]
    assert training_df.shape[0] == portfolio_df.shape[0]
    assert training_df.shape[0] == outcome_df.shape[0]
    assert training_df.application_id.duplicated().sum() == 0

    model_pipeline.fit(training_df.loc[:, X_vars], training_df["default"])
    return model_pipeline


@solid(config_schema={"epoch": int})
def get_epoch():
    epoch = context.solid_config["epoch"]

    return epoch


@solid
def get_applications(context, application_df: pd.DataFrame, epoch: int) -> pd.DataFrame:
    """
    gets applications for new loans from customers
    """

    raw_application_df = get_raw_data()
    new_application_df = raw_application_df.loc[
        (raw_application_df.application_date >= epoch * 100)
        & (raw_application_df.application_date < (epoch + 1) * 100)
    ].copy()
    new_application_df.reset_index(inplace=True, drop=True)
    return application_df.append(
        new_application_df[full_application_col_set]
    ).reset_index(drop=True)


@solid
def choose_business_portfolio(
    context,
    application_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    model_pipeline: Pipeline,
    epoch: int,
    n_loan_budget: int = 100,
) -> pd.DataFrame:
    """
    Decide whom to grant loans to (for profit)

    applications: pd.DataFrame
    model: machine learning model (pipeline) which can be applied to applications, based on training data
    """

    current_application_df = application_df.loc[
        (application_df.application_date >= epoch * 100)
        & (application_df.application_date < (epoch + 1) * 100)
    ]

    current_application_df["est_default_prob"] = pd.DataFrame(
        model_pipeline.predict_proba(
            current_application_df.loc[
                :, ["individual_default_risk", "business_cycle_default_risk"]
            ]
        )
    ).loc[:, 1]

    assert (
        current_application_df.est_default_prob.isna().sum() == 0
    ), "Some estimated default probabilities NaN"

    business_portfolio_df = current_application_df.loc[
        current_application_df["est_default_prob"].rank(method="first") <= n_loan_budget
    ][["application_id"]].reset_index(drop=True)

    business_portfolio_df["funding_probability"] = 1  # TODO: Change this?
    business_portfolio_df["credit_granted"] = True
    return portfolio_df.append(business_portfolio_df[full_portfolio_col_set])


@solid
def choose_research_portfolio(
    context,
    application_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
    model_pipeline: Pipeline,
    active_learning_pipeline: Pipeline,
    epoch: int,
    n_research_budget: int = 20,
) -> pd.DataFrame:
    """
    Decide whom to grant loans to (for research / profit in subsequent rounds)

    business_portfolio: {"application_id", "credit_granted"} as pd.DataFrame
    """
    unfunded_applications = application_df[
        ~application_df.application_id.isin(portfolio_df.application_id.tolist())
        & (application_df.application_date >= epoch * 100)
        & (application_df.application_date < (epoch + 1) * 100)
    ]

    research_portfolio_df = unfunded_applications.sample(n_research_budget)[
        ["application_id"]
    ].reset_index(drop=True)
    research_portfolio_df["portfolio"] = "research"
    research_portfolio_df["credit_granted"] = True
    research_portfolio_df["funding_probability"] = (
        n_research_budget / unfunded_applications.shape[0]
    )
    return portfolio_df.append(research_portfolio_df[full_portfolio_col_set])


@solid
def observe_outcomes(
    context, portfolio_df: pd.DataFrame, outcome_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Observe outcomes to granted credit.
    """
    new_loans = portfolio_df[
        ~portfolio_df.application_id.isin(outcome_df.application_id.tolist()),
        "application_id",
    ].tolist()

    raw_data = get_raw_data()
    return outcome_df.append(
        raw_data[raw_data.application_id.isin(new_loans), full_outcome_col_set]
    )


@pipeline
def active_learning_experiment_credit():
    data = get_historical_data()
    application_df = data["applications"]
    portfolio_df = data["portfolio"]
    outcome_df = data["outcomes"]

    for t in range(10):
        epoch = get_epoch()
        model_pipeline = get_model_pipeline()
        active_learning_pipeline = get_active_learning_pipeline()

        trained_model = train_model(
            application_df,
            portfolio_df,
            outcome_df,
            model_pipeline,
        )

        application_df = get_applications(epoch)

        portfolio_df = choose_business_portfolio(application_df, trained_model, epoch)

        portfolio_df = choose_research_portfolio(
            application_df,
            portfolio_df,
            outcome_df,
            trained_model,
            active_learning_pipeline,
            epoch
        )

        outcome_df = observe_outcomes(portfolio_df, outcome_df)
