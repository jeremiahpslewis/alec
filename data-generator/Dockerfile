FROM julia:1.6
WORKDIR /app

COPY src/Manifest.toml src/Project.toml /app/

RUN julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()' --project=.

# COPY src/ /app/

CMD julia -e "sleep(10000)"
