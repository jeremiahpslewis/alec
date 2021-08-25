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

n_simulations = 3
n_applications_per_period = 1000

function generate_synthetic_data(n_applications_per_period)
    # Delete this line
    # Credit Default Dataset with Business Cycle Effects
    # Assume income, personal default risk, application rate are independent
    # TODO: Explain distribution choices

    simulation_id = string(UUIDs.uuid4())

    loan_data_generator = @model income_based_risk_var, asset_based_risk_var begin
        income_based_risk ~ Normal(0, income_based_risk_var)
        asset_based_risk ~ Normal(0, asset_based_risk_var)
        idiosyncratic_individual_risk ~ Normal(0, 0.1)
        total_default_risk_log_odds = idiosyncratic_individual_risk + income_based_risk + asset_based_risk
        total_default_risk = logistic(total_default_risk_log_odds)
        default ~ Bernoulli(total_default_risk)
    end

    income_based_risk_var = [0.5, 0.5, 0.5, 0.5, 0.5, 2, 2, 2, 2, 2]
    asset_based_risk_var = [2, 2, 2, 2, 2, 0.5, 0.5, 0.5, 0.5, 0.5]

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
