# using Pkg
# Pkg.activate("/app")

using Turing
using StatsPlots
using Distributions
using Parquet
using DataFrames
using Chain
using UUIDs

n_simulations = 100
n_periods = 11
n_applications_per_period = 500
n_applications = n_periods * n_applications_per_period


function generate_synthetic_data(n_applications)
    # Delete this line
    # Credit Default Dataset with Business Cycle Effects
    # Assume income, personal default risk, application rate are independent
    @model function individual_attributes()
        income_based_risk ~ TruncatedNormal(0.3, 0.2, 0, 1)
        assets_based_risk ~ TruncatedNormal(0.7, 0.2, 0, 1)
        income_over_assets_individual_risk_weight ~ TruncatedNormal(0.5, 0.4, 0, 1)
    end

    

    business_cycle_df = DataFrame(
        "application_date" => 1:n_periods,
        "income_over_assets_cycle_risk_weight" => 1 .- 0.9.^(1:n_periods),
    )

    individuals_df = DataFrame(sample(individual_attributes(), NUTS(1000, 0.65), n_applications))
    individuals_df[!, :application_date] .= repeat(1:n_periods, n_applications_per_period)

    portfolio_df = leftjoin(
        individuals_df[!, [:application_date, :income_based_risk, :assets_based_risk, :income_over_assets_individual_risk_weight]],
        business_cycle_df,
        on = :application_date,
    )

    portfolio_df[!, :income_over_assets_gross_risk_weight] =
        (portfolio_df[!, :income_over_assets_individual_risk_weight].^2 .+
        portfolio_df[!, :income_over_assets_cycle_risk_weight].^2).^0.5    
    
    portfolio_df[!, :total_default_risk] = (portfolio_df[!, :income_over_assets_gross_risk_weight] .* portfolio_df[!, :income_based_risk]) .+
    ((1 .- portfolio_df[!, :income_over_assets_gross_risk_weight]) .* portfolio_df[!, :income_based_risk])

    portfolio_df[!, :default] = rand.(Bernoulli.(portfolio_df[!, :total_default_risk]))

    function return_uuid(x)
        return string(UUIDs.uuid4())
    end

    portfolio_df[!, :application_id] = return_uuid.(portfolio_df[!, :total_default_risk])

    simulation_id = string(UUIDs.uuid4())
    transform!(portfolio_df, :application_id => (v -> simulation_id) => :simulation_id)

    portfolio_df[!, :application_date] = portfolio_df[!, :application_date] .+ 2019

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

    df = DataFrame()
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
