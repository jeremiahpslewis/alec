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
    unif ~ MeasureTheory.Uniform()
    age = unif * age_var + age_var
    age_sq = age ^ 2
    idiosyncratic_individual_risk ~ MeasureTheory.Normal(0, (1 / age_var) / 3)
    total_default_risk_log_odds = idiosyncratic_individual_risk + age_sq
    total_default_risk = logistic(total_default_risk_log_odds)
    default ~ MeasureTheory.Bernoulli(total_default_risk)
end

age_var = [0.01, 0.1, 0.5, 3, 4, 4, 4, 4, 4, 4]

business_cycle_df = DataFrame(
    "age_var" => age_var,
)

n_periods = nrow(business_cycle_df)
business_cycle_df[!, "application_date"] = 2020:(2020 + (n_periods - 1))


portfolio_df = DataFrame()
for x in eachrow(business_cycle_df)
    one_cycle_df = DataFrame(rand(loan_data_generator(x.age_var), n_applications_per_period))
    one_cycle_df = @chain one_cycle_df begin
        @transform(:application_date = x.application_date,
                   :default = :default * 1, # convert bool to int
                   :age_var = x.age_var,
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
