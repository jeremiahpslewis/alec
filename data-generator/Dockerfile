FROM julia:1.6

WORKDIR /app

COPY src/Manifest.toml src/Project.toml /app/

RUN apt-get update && apt-get install -y awscli && apt-get clean

RUN julia -e 'using Pkg; Pkg.activate("/app"); Pkg.instantiate(); Pkg.precompile();'

COPY src/ /app/

# RUN JULIA_PROJECT=@. julia /app/credit_generator.jl

CMD julia -e 'sleep(1000)'
