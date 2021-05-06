from dagster import pipeline, solid
import pandas as pd
from sklearn import linear_model


@solid
def gather_historical_data(context) -> pd.DataFrame:
    return pd.DataFrame()


@solid
def fetch_training_data(context, new_data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame()


@solid
def fetch_model_pipeline(_):
    return None


@solid
def train_model(context, training_data: pd.DataFrame, model_pipeline):
    """
    training_data: data collected from previous loans granted, as pd.DataFrame
    model: machine learning model (pipeline) which can be applied to training data
    """
    # Replace this!
    reg = linear_model.LinearRegression()
    reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    return reg


@solid
def collect_applications(_)-> pd.DataFrame:
    """
    Collects applications for new loans from customers
    """
    return pd.DataFrame()


@solid
def choose_business_portfolio(context, applications_df: pd.DataFrame, model)-> pd.DataFrame:
    """
    Decide whom to grant loans to (for profit)

    applications: pd.DataFrame
    model: machine learning model (pipeline) which can be applied to applications, based on training data
    """
    return pd.DataFrame()


@solid
def choose_research_portfolio(
    context, applications_df: pd.DataFrame, business_portfolio_df: pd.DataFrame, model
)-> pd.DataFrame:
    """
    Decide whom to grant loans to (for research / profit in subsequent rounds)

    business_portfolio: {"application_id", "loan_granted"} as pd.DataFrame
    """
    return pd.DataFrame()


@solid
def grant_credit(
    context,
    applications_df: pd.DataFrame,
    business_portfolio_df: pd.DataFrame,
    research_portfolio_df: pd.DataFrame,
)-> pd.DataFrame:
    """
    Grant credit to individuals and observe outcomes
    """
    return pd.DataFrame()


@solid
def observe_outcomes(context, portfolio_df: pd.DataFrame)-> pd.DataFrame:
    """
    Observe outcomes to granted credit.
    """
    return pd.DataFrame()


@pipeline
def active_learning_credit_pipeline():
    historical_data = gather_historical_data()

    for t in range(10):
        training_df = fetch_training_data(historical_data)
        model_pipeline = fetch_model_pipeline()

        trained_model = train_model(training_df, model_pipeline)

        applications_df = collect_applications()

        business_portfolio_df = choose_business_portfolio(
            applications_df, trained_model
        )

        research_portfolio_df = choose_research_portfolio(
            applications_df, business_portfolio_df, trained_model
        )

        portfolio_df = grant_credit(
            applications_df, business_portfolio_df, research_portfolio_df
        )

        historical_data = observe_outcomes(portfolio_df)
