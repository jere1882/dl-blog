#!/bin/bash

# Configuration
FUNCTION_NAME="egyptianArtAnalyzer"
REGION="us-east-2"
PROFILE="egyptian-project"
ACCOUNT_ID=$(aws sts get-caller-identity --profile ${PROFILE} --query Account --output text)
ECR_REPOSITORY="egyptian-art-analyzer"
IMAGE_TAG="latest"
API_GATEWAY_ID=$(aws apigateway get-rest-apis --profile ${PROFILE} --region ${REGION} --query 'items[?name==`egyptianArtAnalyzer-API`].id' --output text)

echo "Building Docker image..."
sudo docker build -t ${ECR_REPOSITORY}:${IMAGE_TAG} .

echo "Creating ECR repository if it doesn't exist..."
aws ecr create-repository --profile ${PROFILE} --repository-name ${ECR_REPOSITORY} --region ${REGION} 2>/dev/null || echo "Repository already exists"

echo "Logging in to ECR..."
aws ecr get-login-password --profile ${PROFILE} --region ${REGION} | sudo docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

echo "Tagging image for ECR..."
sudo docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}

echo "Pushing image to ECR..."
sudo docker push ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}

echo "Creating IAM role if it doesn't exist..."
aws iam create-role \
    --profile ${PROFILE} \
    --role-name lambda-execution-role \
    --assume-role-policy-document '{"Version": "2012-10-17","Statement": [{"Effect": "Allow","Principal": {"Service": "lambda.amazonaws.com"},"Action": "sts:AssumeRole"}]}' \
    --region ${REGION} 2>/dev/null || echo "Role already exists"

echo "Attaching execution policy to role..."
aws iam attach-role-policy \
    --profile ${PROFILE} \
    --role-name lambda-execution-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole \
    --region ${REGION} 2>/dev/null || echo "Policy already attached"

echo "Creating or updating Lambda function..."
aws lambda create-function \
    --profile ${PROFILE} \
    --function-name ${FUNCTION_NAME} \
    --package-type Image \
    --code ImageUri=${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG} \
    --timeout 30 \
    --memory-size 512 \
    --role arn:aws:iam::${ACCOUNT_ID}:role/lambda-execution-role \
    --region ${REGION} 2>/dev/null || \
aws lambda update-function-code \
    --profile ${PROFILE} \
    --function-name ${FUNCTION_NAME} \
    --image-uri ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG} \
    --region ${REGION}

echo "Setting environment variables..."
aws lambda update-function-configuration \
    --profile ${PROFILE} \
    --function-name ${FUNCTION_NAME} \
    --environment Variables="{GOOGLE_API_KEY=${GOOGLE_API_KEY}}" \
    --region ${REGION}

echo "Adding API Gateway permission..."
aws lambda add-permission \
    --profile ${PROFILE} \
    --function-name ${FUNCTION_NAME} \
    --statement-id apigateway-invoke \
    --action lambda:InvokeFunction \
    --principal apigateway.amazonaws.com \
    --source-arn "arn:aws:execute-api:${REGION}:${ACCOUNT_ID}:${API_GATEWAY_ID}/*/*" \
    --region ${REGION} 2>/dev/null || echo "Permission already exists"

echo "Deployment complete!"
echo "Function ARN: arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}"
