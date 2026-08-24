# Scoped permissions for the IAM user that runs `docker push` / `terraform
# apply` for this stack (emotisense-deploy). Replaces the AdministratorAccess
# policy that user was given as a one-time bootstrapping convenience while
# building this by hand via the CLI - everything below is exactly what
# actually got used that session, nothing broader.

data "aws_iam_policy_document" "deploy" {
  # ECR: push/pull images. GetAuthorizationToken can't be scoped to a
  # resource - ECR auth tokens are account/region-wide by AWS's own design.
  statement {
    sid       = "EcrAuth"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid    = "EcrPushPull"
    effect = "Allow"
    actions = [
      "ecr:CreateRepository",
      "ecr:DescribeRepositories",
      "ecr:DescribeImages",
      "ecr:ListImages",
      "ecr:ListTagsForResource",
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
      "ecr:InitiateLayerUpload",
      "ecr:UploadLayerPart",
      "ecr:CompleteLayerUpload",
      "ecr:PutImage",
    ]
    resources = [aws_ecr_repository.api.arn]
  }

  # Lambda: manage this one function and its Function URL.
  statement {
    sid    = "LambdaManageFunction"
    effect = "Allow"
    actions = [
      "lambda:CreateFunction",
      "lambda:GetFunction",
      "lambda:GetFunctionConfiguration",
      "lambda:ListVersionsByFunction",
      "lambda:UpdateFunctionCode",
      "lambda:UpdateFunctionConfiguration",
      "lambda:DeleteFunction",
      "lambda:TagResource",
      "lambda:ListTags",
      "lambda:CreateFunctionUrlConfig",
      "lambda:GetFunctionUrlConfig",
      "lambda:UpdateFunctionUrlConfig",
      "lambda:DeleteFunctionUrlConfig",
      "lambda:AddPermission",
      "lambda:RemovePermission",
      "lambda:GetPolicy",
    ]
    resources = [aws_lambda_function.api.arn]
  }

  # IAM: only for the one execution role this function assumes - not IAM
  # in general. PassRole is scoped further by the service condition, so
  # this can't be used to hand the Lambda role to anything but Lambda.
  statement {
    sid    = "ManageLambdaExecRole"
    effect = "Allow"
    actions = [
      "iam:CreateRole",
      "iam:GetRole",
      "iam:DeleteRole",
      "iam:AttachRolePolicy",
      "iam:DetachRolePolicy",
      "iam:ListAttachedRolePolicies",
      "iam:ListRolePolicies",
      "iam:TagRole",
    ]
    resources = [aws_iam_role.lambda_exec.arn]
  }

  statement {
    sid       = "PassLambdaExecRole"
    effect    = "Allow"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.lambda_exec.arn]

    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["lambda.amazonaws.com"]
    }
  }

  # Self-referential: this policy (and its attachment to the deploy user)
  # are themselves Terraform-managed resources, so the deploy user needs
  # permission to read/update them on every plan/apply - otherwise the
  # very act of running Terraform without admin access breaks once admin
  # access is removed. Scoped to this one policy and this one user only,
  # not IAM policies/users in general.
  statement {
    sid    = "ManageOwnDeployPolicy"
    effect = "Allow"
    actions = [
      "iam:GetPolicy",
      "iam:GetPolicyVersion",
      "iam:ListPolicyVersions",
      "iam:CreatePolicyVersion",
      "iam:DeletePolicyVersion",
      "iam:TagPolicy",
    ]
    # Built directly rather than aws_iam_policy.deploy.arn - referencing
    # the resource here would make the policy document depend on the
    # policy that depends on this very document, a cycle Terraform
    # correctly refuses to plan. The ARN is fully deterministic from the
    # name below anyway, so there's nothing the resource reference would
    # add.
    resources = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:policy/${var.function_name}-deploy-policy"]
  }

  statement {
    sid    = "ManageOwnUserPolicyAttachment"
    effect = "Allow"
    actions = [
      "iam:ListAttachedUserPolicies",
      "iam:AttachUserPolicy",
      "iam:DetachUserPolicy",
    ]
    resources = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:user/${var.deploy_user_name}"]
  }

  # CloudWatch Logs: read-only, for `aws logs filter-log-events` while
  # debugging - and enough to let Lambda's auto-created log group be
  # inspected.
  statement {
    sid    = "ReadLambdaLogs"
    effect = "Allow"
    actions = [
      "logs:DescribeLogGroups",
      "logs:DescribeLogStreams",
      "logs:FilterLogEvents",
      "logs:GetLogEvents",
    ]
    resources = [
      "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/${var.function_name}",
      "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/${var.function_name}:*",
    ]
  }
}

resource "aws_iam_policy" "deploy" {
  name        = "${var.function_name}-deploy-policy"
  description = "Scoped deploy permissions for ${var.function_name} - ECR push, Lambda + its one execution role, log read access. Nothing else."
  policy      = data.aws_iam_policy_document.deploy.json
}

resource "aws_iam_user_policy_attachment" "deploy" {
  user       = var.deploy_user_name
  policy_arn = aws_iam_policy.deploy.arn
}
