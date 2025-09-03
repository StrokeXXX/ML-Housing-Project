resource "aws_sagemaker_model" "housing_model" {
  name               = "housing-model"
  execution_role_arn = aws_iam_role.sagemaker_role.arn
  
  primary_container {
    image = "${aws_ecr_repository.model_repo.repository_url}:latest"
  }
}

resource "aws_sagemaker_endpoint" "housing_endpoint" {
  name                 = "housing-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.housing_config.name
}
