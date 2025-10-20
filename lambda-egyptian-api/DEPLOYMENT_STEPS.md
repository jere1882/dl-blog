# 🚀 Egyptian AI Lens - Deployment Steps

## Prerequisites

- AWS CLI configured with appropriate permissions
- Docker installed and running
- Google API Key for Gemini AI
- AWS Account with Lambda, ECR, and API Gateway permissions

## Step 1: Set Up Environment Variables

```bash
# Set your Google API key
export GOOGLE_API_KEY="your-actual-google-api-key-here"

# Verify it's set
echo $GOOGLE_API_KEY
```

## Step 2: Test Locally (Optional but Recommended)

```bash
# Test the function locally
python test_local.py

# Test the Docker container
sudo docker run --rm -d -p 9000:8080 -e GOOGLE_API_KEY=$GOOGLE_API_KEY --name egyptian-test egyptian-art-analyzer

# Test with curl
curl -X POST http://localhost:9000/2015-03-31/functions/function/invocations \
  -H "Content-Type: application/json" \
  -d '{"httpMethod": "POST", "body": "{\"image\": \"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==\", \"speed\": \"fast\", \"imageType\": \"unknown\"}"}'

# Clean up
sudo docker stop egyptian-test
```

## Step 3: Deploy to AWS Lambda

```bash
# Run the deployment script
./deploy.sh
```

This script will:

1. Build the Docker image
2. Create ECR repository (if it doesn't exist)
3. Push image to ECR
4. Create/update Lambda function
5. Set environment variables

## Step 4: Set Up API Gateway (Manual Steps)

After Lambda deployment, you need to:

1. **Create API Gateway**:

   - Go to AWS Console → API Gateway
   - Create new REST API
   - Create resource and method (POST)
   - Set up Lambda integration

2. **Configure CORS**:

   - Enable CORS for your API
   - Set appropriate headers

3. **Deploy API**:
   - Deploy to a stage (e.g., "prod")
   - Note the API Gateway URL

## Step 5: Test Deployed Function

```bash
# Set your API Gateway URL
export LAMBDA_URL="https://your-api-gateway-url/egyptian-art-analyzer"

# Test the deployed function
python test_lambda.py
```

## Step 6: Update Frontend

Add these environment variables to your `.env.local`:

```bash
NEXT_PUBLIC_LAMBDA_API_URL=https://your-api-gateway-url/egyptian-art-analyzer
NEXT_PUBLIC_API_KEY=your-api-gateway-key-here
```

## Troubleshooting

### Common Issues:

1. **Docker Permission Denied**:

   ```bash
   sudo usermod -aG docker $USER
   # Then logout and login again
   ```

2. **AWS CLI Not Configured**:

   ```bash
   aws configure
   ```

3. **ECR Repository Issues**:

   - Check AWS Console → ECR
   - Ensure repository exists

4. **Lambda Function Timeout**:
   - Increase timeout in Lambda settings
   - Check CloudWatch logs

### Debug Commands:

```bash
# Check AWS CLI
aws sts get-caller-identity

# List Lambda functions
aws lambda list-functions --region us-east-2

# View logs
aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/egyptian-art-analyzer"
```

## Cost Optimization

- **Lambda**: Pay per request and execution time
- **ECR**: Minimal storage cost
- **API Gateway**: Pay per request
- **CloudWatch**: Log storage costs

## Monitoring

- **CloudWatch Logs**: Monitor function execution
- **CloudWatch Metrics**: Track performance
- **X-Ray**: Trace requests (optional)

## Updating the Function

To update after making changes:

```bash
# Make your code changes
# Then run deployment again
./deploy.sh
```

The script will automatically update the existing function.
