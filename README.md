# alec
Active Learning Experiment — Credit (ALEC)


## How to Run

1. Clone Repo
2. Install Garden
3. Install Docker Desktop

```bash
garden run workflow full-pipeline
```

## How to Install Prerequisites on Mac
```bash
brew tap garden-io/garden
brew install garden-cli
```

## How to access the visualization dashboard

```bash
garden dev

# Then click on the link next to 'dash'
```


Note: an AWS account is required to run this project and files must be written to an s3 bucket.


The following environment variables must be specified:

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
S3_BUCKET_NAME
```

Note: the entire project uses multiple (75) simulations to ensure robustness of results. Running the project takes around an hour on a fast machine with ample RAM.
