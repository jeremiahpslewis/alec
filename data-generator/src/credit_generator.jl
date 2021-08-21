# using Pkg
# Pkg.activate("/app")

using Turing
using StatsPlots
using Distributions
using Parquet
using DataFrames
using Chain
using UUIDs
using DataFrameMacros

n_simulations = 7
n_periods = 11
n_applications_per_period = 60
n_applications = n_periods * n_applications_per_period


function generate_synthetic_data(n_applications)
    # Delete this line
    # Credit Default Dataset with Business Cycle Effects
    # Assume income, personal default risk, application rate are independent
    # TODO: Explain distribution choices
    @model function individual_attributes()
        income_based_risk ~ TruncatedNormal(0.3, 0.4, 0, 1)
        asset_based_risk ~ TruncatedNormal(0.7, 0.4, 0, 1)
        idiosyncratic_individual_risk ~ TruncatedNormal(0.5, 0.4, 0, 1)
    end

    business_cycle_df = DataFrame(
        "application_date" => 1:n_periods,
        "income_over_asset_cycle_risk_weight" => [0.01, 0.01, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 0.99, 0.99],
    )

    individuals_df = DataFrame(sample(individual_attributes(), NUTS(1000, 0.65), n_applications))

    portfolio_df = @chain individuals_df begin
        @transform(:application_date = @c repeat(1:n_periods, n_applications_per_period))
        leftjoin(business_cycle_df, on = :application_date)
        @select(:application_date, :income_based_risk, :asset_based_risk, :idiosyncratic_individual_risk, :income_over_asset_cycle_risk_weight)
    end

    simulation_id = string(UUIDs.uuid4())

    portfolio_df = @chain portfolio_df begin

        
        @transform(
            # Weighted average of individual's income risk and asset risk, weight is an increasing function of application_date
            :financial_individual_risk = (:income_over_asset_cycle_risk_weight * :income_based_risk) + 
                ((1 - :income_over_asset_cycle_risk_weight) * :asset_based_risk)
        )
        
    
        @transform(
            # Total default risk is average of financial risk as calculated above and individual's idiosyncratic risk
            :total_default_risk = (:financial_individual_risk + :idiosyncratic_individual_risk) / 2
        )

        # Simulate defaults based on risk
        @transform(
            :default = rand(Bernoulli(:total_default_risk)),
            :application_id = string(UUIDs.uuid4()),
            :simulation_id = simulation_id,
            :application_date = :application_date + 2019
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

function generate_synthetic_data(n_applications, n_simulations)

    # Clear bucket of synthetic data...
    run(`aws s3 rm s3://alec/synthetic_data --recursive`)

    for i = 1:n_simulations
        generate_synthetic_data(n_applications)
    end
end

generate_synthetic_data(n_applications, n_simulations)

# @df portfolio_df boxplot(:default, :total_default_risk)


# tmp_df = @chain portfolio_df begin
#     groupby(:application_date)
#     combine(:default => mean, :default => length)
# end

# @df tmp_df plot(:application_date, :default_mean)

# @df business_cycle_df plot!(:application_date, :business_cycle_default_risk)
