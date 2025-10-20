# 🚀 AWS Lambda Deployment Guide

This guide will help you deploy your Egyptian art analysis API to AWS Lambda and switch your frontend from Vercel to AWS.

## 📋 Prerequisites

1. **AWS CLI** installed and configured with appropriate permissions
2. **Docker** installed and running
3. **Google API Key** for Gemini AI
4. **AWS Account** with permissions to create Lambda functions, ECR repositories, and IAM roles

## 🔧 Step 1: Deploy to AWS Lambda

Since you're using API Gateway with API key authentication, no IAM role setup is needed. The API Gateway handles authentication and authorization.

You can skip the IAM setup and go directly to deployment.

## 🔑 Step 2: Set Environment Variables

Set your Google API key:

```bash
export GOOGLE_API_KEY="your-google-api-key-here"
```

## 🐳 Step 3: Deploy to AWS Lambda

Run the deployment script:

```bash
./deploy.sh
```

This script will:

1. Build the Docker image
2. Create an ECR repository
3. Push the image to ECR
4. Create/update the Lambda function
5. Set environment variables

## 🧪 Step 4: Test the Lambda Function

Test your deployed function:

```bash
# Set your Lambda URL
export LAMBDA_URL="https://your-api-gateway-url/egyptian-art-analyzer"

# Run the test
python test_lambda.py
```

## 🌐 Step 5: Update Your Frontend

Add these environment variables to your `.env.local` file:

```bash
NEXT_PUBLIC_LAMBDA_API_URL=https://your-api-gateway-url/egyptian-art-analyzer
NEXT_PUBLIC_API_KEY=your-api-gateway-key-here
```

Your frontend will now use AWS Lambda exclusively for the Egyptian art analysis API.

## 📊 Step 6: Monitor and Debug

### CloudWatch Logs

Check your Lambda function logs in AWS CloudWatch:

- Go to AWS Console → Lambda → Your Function → Monitor → Logs

### Local Testing

Test the function locally before deployment:

```bash
python test_local.py
```

## 🔧 Step 7: Configuration Options

### Lambda Function Settings

- **Timeout**: 30 seconds
- **Memory**: 512 MB
- **Runtime**: Python 3.11 (Docker container)

### Environment Variables

- `GOOGLE_API_KEY`: Your Google Gemini API key

## 🚨 Step 8: Troubleshooting

### Common Issues

1. **IAM Role Not Found**

   - Run `./setup-iam.sh` again
   - Check that the role name matches in `deploy.sh`

2. **ECR Repository Issues**

   - The script creates the repository automatically
   - Check AWS Console → ECR for the repository

3. **API Gateway CORS**

   - Ensure CORS is configured in API Gateway
   - Check that the Lambda integration is set up correctly

4. **Function Timeout**
   - Increase timeout in Lambda settings
   - Check CloudWatch logs for slow operations

### Debug Commands

```bash
# Check AWS CLI configuration
aws sts get-caller-identity

# List Lambda functions
aws lambda list-functions --region us-east-2

# Get function details
aws lambda get-function --function-name egyptian-art-analyzer --region us-east-2

# View recent logs
aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/egyptian-art-analyzer" --region us-east-2
```

## 📈 Step 9: Performance Optimization

### Lambda Configuration

- **Memory**: 512MB is sufficient for image processing
- **Timeout**: 30 seconds should be enough for Gemini API calls
- **Concurrency**: Set based on expected load

### Cost Optimization

- Lambda pricing: Pay per request and execution time
- ECR storage: Minimal cost for container images
- API Gateway: Pay per request

## 🔄 Step 10: Updating the Function

To update your function after making changes:

```bash
# Make your code changes
# Then run deployment again
./deploy.sh
```

The script will automatically update the existing function.

## 🎉 Success!

Once deployed, your Egyptian art analysis API will be running on AWS Lambda, completely separate from your Vercel deployment. This eliminates the function size issues you were experiencing with Vercel.

Your frontend will now use AWS Lambda exclusively for the Egyptian art analysis functionality!
