# using Pkg
# Pkg.activate("/app")

using Soss
using MeasureTheory
using StatsPlots
using Parquet
using DataFrames
using Chain
using UUIDs
using DataFrameMacros
using GLM

mode = "test"
# mode = "prod"

if mode == "test"
    n_simulations = 3
    n_applications_per_period = 25
elseif mode == "prod"
    n_simulations = 1000
    n_applications_per_period = 25
end

function generate_synthetic_data(n_applications_per_period)
    # Delete this line
    # Credit Default Dataset with Business Cycle Effects
    # Assume income, personal default risk, application rate are independent
    # TODO: Explain distribution choices

    simulation_id = string(UUIDs.uuid4())

    loan_data_generator = @model mean_age begin
        income_based_risk ~ MeasureTheory.Normal(0, income_based_risk_var)
        age ~ MeasureTheory.Uniform(mean_age - 2, mean_age + 2)
        idiosyncratic_individual_risk ~ MeasureTheory.Normal(0, 4)
        total_default_risk_log_odds = idiosyncratic_individual_risk + income_based_risk + age + age^2
        total_default_risk = logistic(total_default_risk_log_odds)
        default ~ MeasureTheory.Bernoulli(total_default_risk)
    end

    mean_age = [25, 40, 55, 70, 85]

    business_cycle_df = DataFrame(
        "mean_age" => mean_age,
    )
    n_periods = nrow(business_cycle_df)
    business_cycle_df[!, "application_date"] = 2020:(2020 + (n_periods - 1))


    portfolio_df = DataFrame()
    for x in eachrow(business_cycle_df)
        one_cycle_df = DataFrame(rand(loan_data_generator(x.mean_age), n_applications_per_period))
        one_cycle_df = @chain one_cycle_df begin
            @transform(:application_date = x.application_date,
                       :default = :default * 1, # convert bool to int
                       :income_based_risk_var = x.income_based_risk_var,
                       :asset_based_risk_var = x.asset_based_risk_var,
                       )
        end
        append!(portfolio_df, one_cycle_df)
    end

    portfolio_df = @chain portfolio_df begin

        # Simulate defaults based on risk
        @transform(
            :application_id = string(UUIDs.uuid4()),
            :simulation_id = simulation_id,
        )
    end

    file_path = "/app/$(simulation_id).parquet"

    @chain portfolio_df begin
        write_parquet(file_path, _)
    end

    run(`aws s3api put-object --bucket alec --key synthetic_data/$(simulation_id).parquet --body $(simulation_id).parquet`)
    rm(file_path)

    return portfolio_df
end

function generate_synthetic_data(n_applications_per_period, n_simulations)

    # Clear bucket of synthetic data...
    run(`aws s3 rm s3://alec/synthetic_data --recursive`)

    for i = 1:n_simulations
        generate_synthetic_data(n_applications_per_period)
    end
end

generate_synthetic_data(n_applications_per_period, n_simulations)

# @df portfolio_df boxplot(:default, :total_default_risk)


# tmp_df = @chain portfolio_df begin
#     groupby(:application_date)
#     combine(:default => mean, :default => length)
# end

# @df tmp_df plot(:application_date, :default_mean)

# @df business_cycle_df plot!(:application_date, :business_cycle_default_risk)

# Analyze two halves of training data (income regime and asset regime)
# Results are that the z-scores and p values for the respective coefficient are very high
# for the dominant variable and quite low for the non-dominant one...

if false
    fm = @formula(default ~ income_based_risk + asset_based_risk)

    portfolio_df_subset = @chain portfolio_df begin
        @subset(:application_date >= 2025)
    end;

    logit = glm(fm, portfolio_df_subset, GLM.Binomial(), LogitLink())

    portfolio_df_subset = @chain portfolio_df begin
        @subset(:application_date < 2025)
    end;

    logit = glm(fm, portfolio_df_subset, GLM.Binomial(), LogitLink())
end
