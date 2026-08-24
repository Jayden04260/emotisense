output "function_url" {
  description = "Public HTTPS endpoint for the API (e.g. append /health or /predict)."
  value       = aws_lambda_function_url.api.function_url
}

output "ecr_repository_url" {
  description = "Push images here (docker tag/push), then update var.image_uri to the new digest."
  value       = aws_ecr_repository.api.repository_url
}

output "lambda_function_arn" {
  value = aws_lambda_function.api.arn
}
