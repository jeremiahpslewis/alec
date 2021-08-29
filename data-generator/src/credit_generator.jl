using Distributions
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
    n_simulations = 1
    n_applications_per_period = 50
elseif mode == "prod"
    n_simulations = 30
    n_applications_per_period = 1000
end


function generate_synthetic_data(n_applications_per_period)
    simulation_id = string(UUIDs.uuid4())

    loan_data_generator = @model age_var begin

        # Age distribution (mean, var) is an increasing function of time, min age of population is set to age_var / 4 for a given time period
        age ~ Distributions.TruncatedNormal(age_var, age_var, age_var / 4, 100)

        # Idiosyncratic individual risk is always zero mean, but the variance is an inverse function of age
        # Older individuals are as such modeled as riskier, but more consistent; younger are safer, but less predictable
        idiosyncratic_individual_risk ~ Distributions.TruncatedNormal(0, 1 / (age * 5), -10, 10)

        # Age is assumed to be associated with increased risk; average risk is located at logistic(-3), approx 4.7% default risk, plus age effects
        total_default_risk_log_odds = idiosyncratic_individual_risk + age - 3

        # Log odds scale is mapped onto zero to one score
        total_default_risk = logistic(total_default_risk_log_odds)

        # Default is a weighted coin flip with weight of 'total_default_risk'
        default ~ MeasureTheory.Bernoulli(total_default_risk)
    end
    
    # Age varible is an increasing function of time, as specified here for each time step
    age_var = [0.25, 0.25, 0.3, 0.4, 0.5, 1, 1.2, 1.5, 2.0, 2.0]
    
    business_cycle_df = DataFrame(
        "age_var" => age_var,
    )

    n_periods = nrow(business_cycle_df)
    business_cycle_df[!, "application_date"] = 2020:(2020 + (n_periods - 1))


    # A separate sample is drawn for each moment in time, based on the age_var for that cross-section
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

    # UUIDs are generated for easier tracking
    portfolio_df = @chain portfolio_df begin
        @transform(
            :application_id = string(UUIDs.uuid4()),
            :simulation_id = simulation_id,
        )
    end

    file_path = "/app/$(simulation_id).parquet"

    # Data is saved in parquet format
    @chain portfolio_df begin
        write_parquet(file_path, _)
    end

    # Data is sent to cloud
    bucket_name = ENV["S3_BUCKET_NAME"]
    run(`aws s3api put-object --bucket $(bucket_name) --key synthetic_data/$(simulation_id).parquet --body $(simulation_id).parquet`)
    rm(file_path)

    return portfolio_df
end

function generate_synthetic_data(n_applications_per_period, n_simulations)
    bucket_name = ENV["S3_BUCKET_NAME"]

    # Clear bucket of synthetic data...
    run(`aws s3 rm s3://$(bucket_name)/synthetic_data --recursive`)

    for i = 1:n_simulations
        generate_synthetic_data(n_applications_per_period)
    end
end

generate_synthetic_data(n_applications_per_period, n_simulations)
