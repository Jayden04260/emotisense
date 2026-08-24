# Terraform - AWS deployment

Codifies the AWS side of the DistilBERT API deployment (see the repo
root's `Dockerfile` and `render.yaml` for the other two deploy targets:
local `docker run` and Render). Manages:

- an ECR repository to hold the Docker image
- a minimal Lambda execution role (CloudWatch Logs only - this function
  talks to the Hugging Face Hub over the internet, not to other AWS
  resources)
- the Lambda function itself (container image, memory/timeout tuned for
  the model-load cold start)
- a public Function URL (not API Gateway - see `variables.tf` for why)
- a scoped IAM policy for the deploy user, replacing the
  `AdministratorAccess` policy used to bootstrap this by hand originally

## What Terraform does *not* do

Terraform manages infrastructure, not the image's contents. Build and
push the image yourself first:

```
docker build --provenance=false --sbom=false -t emotisense-api ..
aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.eu-north-1.amazonaws.com
docker tag emotisense-api:latest <account-id>.dkr.ecr.eu-north-1.amazonaws.com/emotisense-api:latest
docker push <account-id>.dkr.ecr.eu-north-1.amazonaws.com/emotisense-api:latest
```

`--provenance=false --sbom=false` matters: Docker's default build output
includes an attestation manifest that Lambda's container image support
rejects outright.

Then grab the pushed digest (`docker inspect --format='{{index .RepoDigests 0}}' emotisense-api`
or the `docker push` output) and pass it via `-var image_uri=...` (or
edit the default in `variables.tf`) before applying - image_uri is
digest-pinned deliberately, so `terraform apply` never silently
redeploys a different image than the one actually reviewed.

## Usage

```
terraform init
terraform plan
terraform apply
```

## Notable AWS gotchas hit while building this (see repo's interview
notes for the full writeup)

- Public (auth type `NONE`) Function URLs need *two* separate
  `aws_lambda_permission` resources since Oct 2025 - one for
  `lambda:InvokeFunctionUrl`, one for `lambda:InvokeFunction` (via
  `invoked_via_function_url = true`). Missing the second gives a
  persistent 403 that looks like propagation delay but isn't.
- Lambda's filesystem is read-only except `/tmp`. `HF_HOME=/tmp/huggingface`
  redirects the Hugging Face Hub's cache there - without it the app
  crashes on cold start trying to write to a read-only `$HOME`.
