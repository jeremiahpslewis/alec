import pandas as pd
from dagster import pipeline, solid
from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from uuid import uuid4


def get_raw_data():
    raw_df = pd.read_parquet("portfolio_df.parquet")
    return raw_df


@solid
def gather_historical_data(context, epoch_start: int = 0) -> pd.DataFrame:
    historical_df = get_raw_data()
    historical_df = historical_df.loc[
        historical_df.application_date < (epoch_start + 1) * 100
    ].copy()
    return historical_df


@solid
def empty_dataframe(_) -> pd.DataFrame:
    return pd.DataFrame()


@solid
def merge_data(context, historical_data: pd.DataFrame, new_data: pd.DataFrame, applications_df: pd.DataFrame) -> pd.DataFrame:
    return historical_data.append(new_data)


@solid
def fetch_model_pipeline(_) -> Pipeline:
    column_trans = ColumnTransformer(
        [
            ("individual_default_risk", StandardScaler(), ["individual_default_risk"]),
            (
                "business_cycle_default_risk",
                StandardScaler(),
                ["business_cycle_default_risk"],
            ),
        ],
        remainder="drop",
    )
    model_pipeline = make_pipeline(column_trans, linear_model.LogisticRegression())
    return model_pipeline


@solid
def train_model(
    context, training_df: pd.DataFrame, model_pipeline: Pipeline
) -> Pipeline:
    """
    training_data: data collected from previous loans granted, as pd.DataFrame
    model: machine learning model (pipeline) which can be applied to training data
    """
    # Replace this!
    model_pipeline.fit(training_df, training_df["default"])
    return model_pipeline


@solid
def collect_applications(context, epoch: int) -> pd.DataFrame:
    """
    Collects applications for new loans from customers
    """

    applications_df = get_raw_data()
    applications_df = applications_df.loc[
        (applications_df.application_date >= epoch * 100)
        & (applications_df.application_date < (epoch + 1) * 100)
    ].copy()
    applications_df.reset_index(inplace=True, drop=True)
    return applications_df


@solid
def choose_business_portfolio(
    context,
    applications_df: pd.DataFrame,
    model_pipeline: Pipeline,
    n_loan_budget: int = 100,
) -> pd.DataFrame:
    """
    Decide whom to grant loans to (for profit)

    applications: pd.DataFrame
    model: machine learning model (pipeline) which can be applied to applications, based on training data
    """

    applications_df["est_default_prob"] = pd.DataFrame(
        model_pipeline.predict_proba(applications_df)
    ).loc[:, 1]

    assert (
        applications_df.est_default_prob.isna().sum() == 0
    ), "Some estimated default probabilities NaN"

    business_portfolio_df = applications_df.loc[
        applications_df["est_default_prob"].rank(method="first") <= n_loan_budget
    ][["application_id"]].reset_index(drop=True)

    return business_portfolio_df


@solid
def choose_research_portfolio(
    context,
    applications_df: pd.DataFrame,
    business_portfolio_df: pd.DataFrame,
    model_pipeline: Pipeline,
    n_research_budget: int = 20,
) -> pd.DataFrame:
    """
    Decide whom to grant loans to (for research / profit in subsequent rounds)

    business_portfolio: {"application_id", "loan_granted"} as pd.DataFrame
    """
    research_portfolio_df =  applications_df[
        ~applications_df.application_id.isin(
            business_portfolio_df.application_id.tolist()
        )
    ].sample(n_research_budget)[["application_id"]].reset_index(drop=True)

    return research_portfolio_df


@solid
def grant_credit(
    context,
    applications_df: pd.DataFrame,
    business_portfolio_df: pd.DataFrame,
    research_portfolio_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Grant credit to individuals and observe outcomes
    """
    applications_df["portfolio"] = "rejected"
    applications_df.loc[applications_df.application_id.isin(business_portfolio_df.application_id.tolist()), "portfolio"] = "business"
    applications_df.loc[applications_df.application_id.isin(research_portfolio_df.application_id.tolist()), "portfolio"] = "research"

    applications_df["credit_granted"] = applications_df.portfolio != "rejected"
    portfolio_df = applications_df[["application_id", "portfolio", "credit_granted"]]
    return portfolio_df


@solid
def observe_outcomes(context, applications_df: pd.DataFrame, portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """
    Observe outcomes to granted credit.
    """

    raw_data = applications_df[["application_id", "default"]]
    portfolio_outcomes_df = pd.merge(portfolio_df, raw_data, on="application_id", how="left")
    return portfolio_outcomes_df


@pipeline
def active_learning_credit_pipeline():
    historical_data = gather_historical_data()

    for t in range(10):
        model_pipeline = fetch_model_pipeline()

        trained_model = train_model(historical_data, model_pipeline)

        applications_df = collect_applications(t)

        business_portfolio_df = choose_business_portfolio(
            applications_df, trained_model
        )

        research_portfolio_df = choose_research_portfolio(
            applications_df, business_portfolio_df, trained_model
        )

        portfolio_df = grant_credit(
            applications_df, business_portfolio_df, research_portfolio_df
        )

        portfolio_outcomes_df = observe_outcomes(applications_df, portfolio_df)
        historical_data = merge_data(historical_data, portfolio_outcomes_df, applications_df)
