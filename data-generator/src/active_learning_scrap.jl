using Soss
using MeasureTheory
using StatsPlots
using Parquet
using DataFrames
using Chain
using UUIDs
using DataFrameMacros
using GLM

n_applications_per_period = 10000


loan_data_generator = @model age_var begin
    idiosyncratic_individual_risk ~ MeasureTheory.Normal(0, 2)
    unif ~ MeasureTheory.Uniform()
    age = unif * age_var + age_var
    total_default_risk_log_odds = idiosyncratic_individual_risk + age^2
    total_default_risk = logistic(total_default_risk_log_odds)
    default ~ MeasureTheory.Bernoulli(total_default_risk)
end

age_var = [0.01, 0.1, 0.5, 3, 4, 4, 4, 4, 4, 4]
income_based_risk_var = [0.1, 0.1, 0.1, 0.1, 0.1, 4, 4, 4, 4, 4]
asset_based_risk_var = [4, 4, 4, 4, 4, 0.1, 0.1, 0.1, 0.1, 0.1]

business_cycle_df = DataFrame(
    "age_var" => age_var,
    "income_based_risk_var" => income_based_risk_var,
    "asset_based_risk_var" => asset_based_risk_var,
)

n_periods = nrow(business_cycle_df)
business_cycle_df[!, "application_date"] = 2020:(2020 + (n_periods - 1))


portfolio_df = DataFrame()
for x in eachrow(business_cycle_df)
    one_cycle_df = DataFrame(rand(loan_data_generator(x.age_var, x.income_based_risk_var, x.asset_based_risk_var), n_applications_per_period))
    one_cycle_df = @chain one_cycle_df begin
        @transform(:application_date = x.application_date,
                   :default = :default * 1, # convert bool to int
                   :income_based_risk_var = x.income_based_risk_var,
                   :asset_based_risk_var = x.asset_based_risk_var,
                   )
    end
    append!(portfolio_df, one_cycle_df)
end


fm = @formula(default ~ age^2)

portfolio_df_subset = @chain portfolio_df begin
    @subset(:application_date >= 2020)
end;

logit = glm(fm, portfolio_df_subset, GLM.Binomial(), LogitLink())

# Mean squared error
mean(abs.(portfolio_df_subset[!, :total_default_risk] - GLM.predict(logit, portfolio_df_subset)).^2)

portfolio_df_subset = @chain portfolio_df begin
    @subset(:application_date == 2020)
end;

logit = glm(fm, portfolio_df_subset, GLM.Binomial(), LogitLink())

# Mean squared error
mean(abs.(portfolio_df_subset[!, :total_default_risk] - GLM.predict(logit, portfolio_df_subset)).^2)
