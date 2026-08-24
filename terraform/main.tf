data "aws_caller_identity" "current" {}

# ---------------------------------------------------------------------------
# ECR - holds the Docker image (see ../Dockerfile). Image scanning on push
# catches known CVEs in the base image / dependencies for free.
# ---------------------------------------------------------------------------
resource "aws_ecr_repository" "api" {
  name = var.function_name

  image_scanning_configuration {
    scan_on_push = true
  }
}

# ---------------------------------------------------------------------------
# Lambda execution role - deliberately minimal. This function only talks to
# the Hugging Face Hub over the public internet at startup; it doesn't read
# or write any other AWS resource, so CloudWatch Logs access is genuinely
# everything it needs.
# ---------------------------------------------------------------------------
data "aws_iam_policy_document" "lambda_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda_exec" {
  # Matches the role's real, already-live name exactly (created by hand
  # as "emotisense-lambda-role", not "<function_name>-lambda-role") -
  # deriving this from var.function_name would force a destroy/recreate
  # of a role a running Lambda function depends on, purely over a naming
  # convention mismatch that doesn't reflect anything actually wrong.
  name               = "emotisense-lambda-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
}

resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# ---------------------------------------------------------------------------
# The function itself. Runs the same image as Render (see ../Dockerfile) -
# the AWS Lambda Web Adapter layer baked into that image is what lets a
# plain uvicorn/FastAPI server run on Lambda unmodified.
# ---------------------------------------------------------------------------
resource "aws_lambda_function" "api" {
  function_name = var.function_name
  role          = aws_iam_role.lambda_exec.arn

  package_type = "Image"
  image_uri    = var.image_uri

  memory_size = var.memory_size
  timeout     = var.timeout_seconds

  environment {
    variables = {
      MODEL_SOURCE = var.model_source
      # Lambda's filesystem is read-only except /tmp - huggingface_hub
      # defaults to caching under $HOME, which isn't writable here.
      HF_HOME = "/tmp/huggingface"
    }
  }

  depends_on = [aws_iam_role_policy_attachment.lambda_basic_execution]
}

# ---------------------------------------------------------------------------
# Public Function URL - deliberately not API Gateway (API Gateway's free
# tier is only 12 months; Function URLs are free indefinitely, same as
# Lambda's own Always Free tier).
# ---------------------------------------------------------------------------
resource "aws_lambda_function_url" "api" {
  function_name      = aws_lambda_function.api.function_name
  authorization_type = "NONE"
  invoke_mode        = "BUFFERED"
}

# Since October 2025, a public (NONE auth) Function URL needs BOTH of the
# statements below - lambda:InvokeFunctionUrl alone still 403s.
resource "aws_lambda_permission" "url_invoke_function_url" {
  statement_id           = "FunctionURLAllowPublicAccess"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.api.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}


# The second required statement (lambda:InvokeFunction, gated by the
# lambda:InvokedViaFunctionUrl condition) can't be expressed through
# aws_lambda_permission: as of provider v5.100, its schema has no
# argument for that condition key at all (only function_url_auth_type,
# which covers the InvokeFunctionUrl statement above, not this one) -
# confirmed against `terraform providers schema -json` directly, not
# assumed. The AWS CLI exposes it as a distinct --invoked-via-function-url
# flag on the same AddPermission API call the provider wraps, so this is
# a real provider gap, not a config mistake. Shelling out to the CLI for
# just this one statement keeps `terraform apply` self-sufficient instead
# of leaving a manual post-apply step.
resource "terraform_data" "url_invoke_function_permission" {
  # Destroy-time provisioners may only reference `self` - not other
  # resources or variables - so the region travels alongside the function
  # name here instead of being read from var.aws_region directly below.
  triggers_replace = [aws_lambda_function.api.function_name, var.aws_region]

  # Single-line commands deliberately - local-exec runs through cmd.exe on
  # Windows, which doesn't understand Unix-style backslash line
  # continuations (they broke argument parsing entirely the first time
  # this was written with them).
  provisioner "local-exec" {
    command = "aws lambda add-permission --function-name ${self.triggers_replace[0]} --statement-id UrlPolicyInvokeFunction --action lambda:InvokeFunction --principal \"*\" --invoked-via-function-url --region ${self.triggers_replace[1]} || exit 0"
  }

  provisioner "local-exec" {
    when    = destroy
    command = "aws lambda remove-permission --function-name ${self.triggers_replace[0]} --statement-id UrlPolicyInvokeFunction --region ${self.triggers_replace[1]} || exit 0"
  }
}
