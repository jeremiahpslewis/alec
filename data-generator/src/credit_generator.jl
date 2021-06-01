# using Pkg
# Pkg.activate("/app")

using Turing
using StatsPlots
using Distributions
using Parquet
using DataFrames
using Chain
using UUIDs

n_applications = 5000
n_simulations = 500

function generate_synthetic_data(n_applications)
    # Delete this line
    # Credit Default Dataset with Business Cycle Effects
    # Assume income, personal default risk, application rate are independent
    @model function individual_attributes()
        income_monthly ~ Gamma(10, 150)
        individual_default_risk ~ Beta(2, 30)
    end

    # Assume business-cycle risk is a function of date and white noise
    @model function business_cycle()
        shift ~ Normal(50, 200)
        period ~ Normal(365, 30)
    end

    business_cycle_params = DataFrame(sample(business_cycle(), NUTS(1000, 0.65), 1))
    period = business_cycle_params[1, :period]
    shift = business_cycle_params[1, :shift]

    business_cycle_t(t) = 0.5 * sin((t - shift) / (period / (2 * pi))) + 0.5
    # business_cycle_with_noise = business_cycle_t.(1:(365*2)) .* rand(TruncatedNormal(0.95, 0.1, 0.01, 1), 365*2)
    business_cycle_with_noise =
        business_cycle_t.((1:10) * 365) .^ rand(Truncated(Poisson(5), 1, 5), 10)
    business_cycle_default_risk = (rand(Uniform(0.1, 0.5)) .* business_cycle_with_noise)
    # plot(1:10, business_cycle_risk)

    business_cycle_df = DataFrame(
        "application_date" => 1:10,
        "business_cycle_default_risk" => business_cycle_default_risk,
    )
    individuals_df = DataFrame(sample(individual_attributes(), NUTS(1000, 0.65), n_applications))
    individuals_df[!, :application_date] .= repeat(1:10, Int(ceil(n_applications / 10)))[1:n_applications]

    portfolio_df = leftjoin(
        individuals_df[!, [:application_date, :individual_default_risk]],
        business_cycle_df,
        on = :application_date,
    )

    portfolio_df[!, :total_default_risk] =
        clamp.(
            portfolio_df[!, :individual_default_risk] .+
            portfolio_df[!, :business_cycle_default_risk],
            0,
            1,
        )

    portfolio_df[!, :default] = rand.(Bernoulli.(portfolio_df[!, :total_default_risk]))

    function return_uuid(x)
        return string(UUIDs.uuid4())
    end

    portfolio_df[!, :application_id] = return_uuid.(portfolio_df[!, :total_default_risk])

    simulation_id = string(UUIDs.uuid4())
    transform!(portfolio_df, :application_id => (v -> simulation_id) => :simulation_id)

    portfolio_df[!, :application_date] = portfolio_df[!, :application_date] .÷ 100

    return portfolio_df
end

function generate_synthetic_data(n_applications, n_simulations)

    df = DataFrame()
    for i = 1:n_simulations
        @chain generate_synthetic_data(n_applications) begin
            write_parquet("/app/synthetic_data_$(string(UUIDs.uuid4())).parquet", _)
        end
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
