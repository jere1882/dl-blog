# Egyptian Art Analyzer - AWS Lambda

This is the AWS Lambda version of the Egyptian art analysis API, converted from the Vercel function.

## Prerequisites

1. **AWS CLI configured** with appropriate permissions
2. **Docker** installed and running
3. **Google API Key** for Gemini AI
4. **IAM Role** with Lambda execution permissions

## Setup

### 1. Create IAM Role

Create an IAM role with the following policies:

- `AWSLambdaBasicExecutionRole`
- `AWSLambdaVPCAccessExecutionRole` (if needed)

### 2. Set Environment Variables

```bash
export GOOGLE_API_KEY="your-google-api-key-here"
```

### 3. Deploy

```bash
cd lambda-egyptian-api
./deploy.sh
```

## API Usage

The Lambda function expects a JSON POST request with the following structure:

```json
{
  "image": "base64-encoded-image-data",
  "speed": "fast", // optional: "fast", "regular", "super-fast"
  "imageType": "unknown" // optional: "tomb", "temple", "other", "unknown"
}
```

### Example Request

```bash
curl -X POST https://your-api-gateway-url/egyptian-art-analyzer \
  -H "Content-Type: application/json" \
  -d '{
    "image": "iVBORw0KGgoAAAANSUhEUgAA...",
    "speed": "fast",
    "imageType": "tomb"
  }'
```

### Response Format

```json
{
  "translation": "Ancient text translation...",
  "characters": [
    {
      "character_name": "Osiris",
      "reasoning": "Identified by characteristic crown and staff",
      "description": "God of the afterlife",
      "location": "center of the image"
    }
  ],
  "location": "Valley of the Kings, Tomb of Tutankhamun",
  "processing_time": "Analysis completed in 2.34s",
  "interesting_detail": "The hieroglyphs show the royal cartouche...",
  "date": "New Kingdom"
}
```

## Configuration

- **Timeout**: 30 seconds
- **Memory**: 512 MB
- **Runtime**: Python 3.11
- **Architecture**: x86_64

## Troubleshooting

1. **Check CloudWatch Logs** for detailed error messages
2. **Verify API Gateway** is configured to pass through the request body
3. **Ensure CORS** is properly configured if calling from a web browser
4. **Check IAM permissions** for the Lambda execution role

## Local Testing

You can test the function locally using the AWS SAM CLI or by running the Docker container directly:

```bash
docker build -t egyptian-art-analyzer .
docker run -p 9000:8080 -e GOOGLE_API_KEY=your-key egyptian-art-analyzer
```

Then test with:

```bash
curl -X POST http://localhost:9000/2015-03-31/functions/function/invocations \
  -d '{"httpMethod": "POST", "body": "{\"image\": \"test\"}"}'
```
