variable "aws_region" {
  description = "AWS region the API is deployed in."
  type        = string
  default     = "eu-north-1"
}

variable "function_name" {
  description = "Name shared by the Lambda function and ECR repository."
  type        = string
  default     = "emotisense-api"
}

variable "image_uri" {
  description = <<-EOT
    Full ECR image URI (digest-pinned, not `:latest`) the Lambda function
    runs. Terraform manages the function's infrastructure, not the image
    itself - build and push it separately (see terraform/README.md), then
    pass the resulting digest here. Pinning to a digest (rather than a
    mutable tag) means a `terraform apply` never silently redeploys a
    different image than the one actually reviewed.
  EOT
  type        = string
  default     = "473247068720.dkr.ecr.eu-north-1.amazonaws.com/emotisense-api@sha256:769163730b5e4ee902e1911407758fdb8aac1ac3e590e6fb02436b49c7686faf"
}

variable "model_source" {
  description = "Hugging Face Hub repo id the API loads weights from at startup."
  type        = string
  default     = "Jayden04260/emotisense-distilbert"
}

variable "memory_size" {
  description = <<-EOT
    Lambda memory in MB. Deliberately generous (not the minimum that
    fits): Lambda allocates CPU proportional to memory, and this
    function's cold start is dominated by loading the DistilBERT model,
    not by RAM pressure - more memory here means a faster cold start,
    not just more headroom.
  EOT
  type        = number
  default     = 3008
}

variable "timeout_seconds" {
  description = "Lambda timeout. Must comfortably exceed worst-case cold-start model load time (~26s measured)."
  type        = number
  default     = 120
}

variable "deploy_user_name" {
  description = "IAM user this stack grants scoped deploy permissions to (replaces a manually-attached AdministratorAccess policy used to bootstrap this the first time)."
  type        = string
  default     = "emotisense-deploy"
}
