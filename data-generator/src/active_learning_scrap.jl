using Soss
using MeasureTheory
using StatsPlots
using Parquet
using DataFrames
using Chain
using UUIDs
using DataFrameMacros
using GLM

n_applications_per_period = 250


loan_data_generator = @model income_based_risk_var, asset_based_risk_var begin
    income_based_risk ~ MeasureTheory.Normal(0, income_based_risk_var)
    asset_based_risk ~ MeasureTheory.Normal(0, asset_based_risk_var)
    idiosyncratic_individual_risk ~ MeasureTheory.Normal(0, 4)
    total_default_risk_log_odds = idiosyncratic_individual_risk + income_based_risk + asset_based_risk
    total_default_risk = logistic(total_default_risk_log_odds)
    default ~ MeasureTheory.Bernoulli(total_default_risk)
end

income_based_risk_var = [0.1, 0.1, 0.1, 0.1, 0.1, 4, 4, 4, 4, 4]
asset_based_risk_var = [4, 4, 4, 4, 4, 0.1, 0.1, 0.1, 0.1, 0.1]

business_cycle_df = DataFrame(
    "income_based_risk_var" => income_based_risk_var,
    "asset_based_risk_var" => asset_based_risk_var,
)

n_periods = nrow(business_cycle_df)
business_cycle_df[!, "application_date"] = 2020:(2020 + (n_periods - 1))


portfolio_df = DataFrame()
for x in eachrow(business_cycle_df)
    one_cycle_df = DataFrame(rand(loan_data_generator(x.income_based_risk_var, x.asset_based_risk_var), n_applications_per_period))
    one_cycle_df = @chain one_cycle_df begin
        @transform(:application_date = x.application_date,
                   :default = :default * 1, # convert bool to int
                   :income_based_risk_var = x.income_based_risk_var,
                   :asset_based_risk_var = x.asset_based_risk_var,
                   )
    end
    append!(portfolio_df, one_cycle_df)
end


fm = @formula(default ~ income_based_risk + asset_based_risk)

portfolio_df_subset = @chain portfolio_df begin
    @subset(:application_date == 2020)
end;

logit = glm(fm, portfolio_df_subset, GLM.Binomial(), LogitLink())

# Mean squared error
mean(abs.(portfolio_df_subset[!, :total_default_risk] - GLM.predict(logit, portfolio_df_subset)).^2)
