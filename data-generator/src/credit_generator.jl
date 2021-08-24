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

n_simulations = 7
n_periods = 11
n_applications_per_period = 60

function generate_synthetic_data(n_applications_per_period, n_periods)
    # Delete this line
    # Credit Default Dataset with Business Cycle Effects
    # Assume income, personal default risk, application rate are independent
    # TODO: Explain distribution choices

    simulation_id = string(UUIDs.uuid4())

    loan_data_generator = @model income_over_asset_cycle_risk_weight begin
        income_based_risk ~ Normal(0, 1)
        asset_based_risk ~ Normal(0, 1)
        idiosyncratic_individual_risk ~ Normal(0, 1)
        logitp = idiosyncratic_individual_risk + income_over_asset_cycle_risk_weight * income_based_risk + (1 - income_over_asset_cycle_risk_weight) * asset_based_risk
        total_default_risk = logistic(logitp)
        z ~ Bernoulli(logistic(logitp))
    end

    n_applications_per_period = 10

    business_cycle_df = DataFrame(
        "application_date" => 2020:(2020 + n_periods),
        "income_over_asset_cycle_risk_weight" => [0.01, 0.01, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 0.99, 0.99],
    )

    portfolio_df = DataFrame()
    for x in eachrow(business_cycle_df)
        one_cycle_df = DataFrame(rand(loan_data_generator(x.income_over_asset_cycle_risk_weight), n_applications_per_period))
        one_cycle_df = @chain one_cycle_df begin
            @transform(:application_date = x.application_date, :income_over_asset_cycle_risk_weight = x.income_over_asset_cycle_risk_weight)
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

function generate_synthetic_data(n_applications_per_period, n_periods, n_simulations)

    # Clear bucket of synthetic data...
    run(`aws s3 rm s3://alec/synthetic_data --recursive`)

    for i = 1:n_simulations
        generate_synthetic_data(n_applications_per_period, n_periods)
    end
end

generate_synthetic_data(n_applications_per_period, n_periods, n_simulations)

# @df portfolio_df boxplot(:default, :total_default_risk)


# tmp_df = @chain portfolio_df begin
#     groupby(:application_date)
#     combine(:default => mean, :default => length)
# end

# @df tmp_df plot(:application_date, :default_mean)

# @df business_cycle_df plot!(:application_date, :business_cycle_default_risk)
