# alec
Active Learning Experiment — Credit (ALEC)


## How to Run

1. Clone Repo
2. Install Garden
3. Install Docker Desktop

```bash
garden run workflow gen-data
```

```bash
brew tap garden-io/garden
brew install garden-cli

garden dev
garden exec dagster bash
```


Note: an AWS account is required to run this project and files must be written to an s3 bucket.


The following environment variables must be specified:

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```
