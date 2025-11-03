---
tags:
  - LLM
aliases:
publish: true
slug: deploying-your-model-in-aws-lambda
title: Deploying your ML model in AWS lambda
description: Step by step guide
date: 2025-09-14
image: /thumbnails/pick_architecure.jpeg
---
This guide explains how to deploy your ML model entirely from scratch on an AWS lambda. I will present two examples:

* A simpler lambda that calls Gemini to generate an analysis of an input image of Egyptian art
* A slightly more complex lambda that runs a custom-trained computer vision model

Across the post, you will notice text boxes with a 💡 icon. These are definitions and clarifications intended for people who are new to the AWS ecosystem.

# Preamble: Is a Lambda suitable for my deployment?

Lambdas are caracterized by:
* serverless
* event-driven
* strict runtime and resource limites

Therefore, it's only suitable for:

* Lightweight models with fast inference: e.g. logistic regression, small random forest, lightweight neural network
* Low-throughput: suitable if you don't want to pay for an always-on EC2 or Sagemaker endpoint. Lambdas are charged only by invocation, and scale automatically to handle bursts. Suitable if requests arrive sporadically (e.g. a few times an hour/day)
* Cheaper for spiky or low traffic compared to paying for dedicated instances
* Suitable also if you don't have heavy data transformations, preprocessing and posprocessing.

In other words, a lambda is NOT suitable for:
* Large model sizes: container images are limited to 10GB, unzipped files are limited to 250 MB. Bigger models need ECS/EC2/SageMaker
* Long inference time: Lambda is best for inference under 1-3 seconds, otherwise cold start + scaling overhead make it inefficient. Lambdas have a hard timeout at 15 minutes.
* High throughput / low latency: lambdas cold start, which cause latency 
* GPU requirements: Lambdas do not support GPU acceleration.

As of 2025:
* Lambdas have memory configurable up to 10 GB
* Lambdas allocate CPUs proportional to memory, up to  6 vCPU
* No GPU available
* Up to 10 GB ephemeral storage
* Deployment size: up to 250 MB uncompressed (zip deployment) or 10 GB (container image)

To sum up:
* EC2: Classic VM approach. You rent a virtual machine, install pytorch, your model and dependencies, and run an API server (like FastAPI) to serve requests.
	* full control over env and hardware
	* instance runs 24/7, you pay even when idle
	* you need to handle scaling, security patche, etc
* AWS Lambda "serverless" approach
	* pay per request + compute time
	* no server management
	* available 24/7 without a bill
	* cold start time
	* model size and compute limits
* Amazon SageMaker 
	* handles packaging, deployment and scaling
	* it's designed for ML, no need to manually build an API
	* you pay for endpoints even if idle
# Example 1: Egyptian image analyzer via Gemini

*Fill later: overview of steps*
## Step 1: Create an AWS account (if you don't have one), IAM user and attach policies

### Create a root user
* Go to https://signin.aws.amazon.com/signup?request_type=register and create a new AWS account. You'll be prompted to add a bunch of personal information.
* Set up billing - you get $200 for free, and you can set up a credit card already as well.

This will create a root user for your new AWS account. 

>💡 **about AWS root user**
> The root user should not be used in actual projects, that's why crease specific IAM users with restricted sets of permissions for each project. AWS recommends that the root account never uses access keys, except for very rare cases. Instead, you create an **IAM user** (or IAM role) and give _that_ user access keys. Why? If the keys leak, the attacker doesn’t get full root powers. The **root user** (the email + password you used to create the AWS account) is _too powerful_ and dangerous for everyday use.
> 
> Analogy: Imagine AWS is a giant office building:
> - **Root user** = the building owner with a master key (opens everything, including the vault).
> - **IAM users** = employees with keycards that open only the doors they need.

### Create a new IAM user

>💡 **about IAM users**
>An IAM (Identity and Access Management) user is a digital entity you create inside an AWS account. It represents a person or an application that needs to interact with AWS. 
>- Think of it as making an extra login for your cloud account, but only with the permissions you grant, not full "root" control
>- An IAM user has: a username within the AWS account, a console password, access keys (ID+secret). It may have assigned permissions via two methods: 
	1. Directly attached policies (e.g. AmazonS3ReadOnlyAccess)
	2. Membership in IAM groups policies
> - An IAM user is always scoped within the AWS account, it's not global like the root login.
		
* Login into AWS Management Console https://console.aws.amazon.co
* Go to IAM (Identity and Access Management) - Left menu -> Users -> Add user
	* Make sure programatic access is enabled
	* Attach policies to the newly created user (permissions)  **Add permissions** → **Attach existing policies directly**.****:
		- `AWSLambda_FullAccess` → to create/manage Lambda functions    
		- `IAMFullAccess` → to create roles for Lambda
		- `AmazonAPIGatewayAdministrator` → if you want to expose Lambda via API Gateway
		- `CloudWatchLogsFullAccess` → so Lambda can write logs
* Ultimately you get an AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY


![[Pasted image 20250921110242.png]]

### Set up aws-cli

Install `aws-cli` and configure credentials:

```
sudo apt install awscli -y   # (Linux example)
aws configure --profile egyptian-project
```

* you will be prompted to input the access key, secret access key, etc. Use the ones of the new IAM user here.
* this creates entries in `~/.aws/credentials` and `~/.aws/config`
* Run `aws sts get-caller-identity --profile egyptian-project` -> should show your user id, account, etc.

>💡 **AWS users and their permissions**
> AWS has two ways to create users:
> 1. **(classic) IAM users**: Create long lived IAM users with access keys and passwords (as described above)  =>  **IAM -> Users -> Add users**
> 2. **IAM Identity Center**: (Formerly AWS SSO) Secure login w/ MFA and temporary credentials. This allows login via `aws sso login --profile myprofile,` and AWS gives you temporary credentials, which is safer than long-lived keys 
>    
>These are two different sets of users: the **users you create in Identity Center don’t show up in classic IAM**, and vice versa.
>
>For this demo, we are going to use classic users.
## Step 2: Create the Lambda function

Let's create the lambda function itself:

Go to the main AWS console https://console.aws.amazon.com/ and log in with the IAM user credentials (username + password)
	1. account id: tells which AWS account the IAM user belongs to (12-digit number that you can find in the AWS root account).
	2. IAM username: The name you gave to your IAM user.


Once you managed to log in, go to **Lambda** > Create Function

![[Pasted image 20250928132849.png]]
 Select "Author from Scratch", so that we can create a first version that simply returns a "hello world"

> 💡 **Lambda creation options**
> Three options are available:
> 1. **Author from Scratch**, where we'll write the function code and configure it manually. We'll pick this option for a first version.
> 2. **Use a Blueprint:** pre-configured templates for e.g. webhook handling, event processing, etc.
> 3. **Container image:** Package the lambda function as a Docker container and deploy it to lambda - we'll switch to this option in the second version.

Give the new lambda a representative name and select a runtime - For my use case, Python 3.9 was ok. The runtime refers to the programming language and environment in which the Lambda function will execute.

Permissions: Lambda needs an **execution role** to run. Create a new role from AWS policy templates:
		1. Role name: `lambda-execution-role`
		2. Policy templates: Basic Lambda permissions (this attaches`AWSLambdaBasicExecutionRole`)

>💡 **Execution Roles**
>An **execution role** is an **AWS Identity and Access Management (IAM) role** that grants **AWS Lambda** the necessary permissions to interact with other AWS services on your behalf. Lambda functions don't run in isolation; they often need access to other AWS resources
>`AWSLambdaBasicExecutionRole`:  This is a predefined AWS managed policy that grants Lambda the basic permissions to write logs to CloudWatch Logs (for logging function output, errors, etc.) and basic permissions required to allow the Lambda function to be executed.

For a first version, let's just edit the **function code** in the inline editor of the console (this is the **Code** tab), so that it just returns a Hello World. We'll soon replace this by the actual ML magic.

```python
import json

def lambda_handler(event, context):
	# Example: read JSON body from POST request
	body = json.loads(event.get("body", "{}"))
	name = body.get("name", "stranger") # default if no name provided
	
	response_text = f"Hello {name}!"
	
	return {
		"statusCode": 200,
		"headers": {"Content-Type": "application/json"}	
		"body": json.dumps({"message": response_text})	
		}
```

![[Pasted image 20250929003818.png]]

Click **deploy** to save the inline code.

## Step 3: Expose the Lambda via API Gateway

Let's expose the lambda via **API Gateway**, because we want the lambda to be called via **HTTP**. From the lambdas home screen, go to **Configuration → Triggers → Add trigger → API Gateway**:

![[Pasted image 20250929004501.png]]


Create a new REST API. Set to **open security** for now, then we'll add security.
![[Pasted image 20250929000621.png]]
We have a public endpoint now! something like `https://xxxxxx.execute-api.us-east-1.amazonaws.com/default/egyptianArtAnalyzer

![[Pasted image 20250929004711.png]]

Let's test the setup - replace the text in the message with your own name and the url with your API's url:

```bash
$ curl -X POST https://XXXXXX.execute-api.us-east-2.amazonaws.com/default/egyptianArtAnalyzer   -H "Content-Type: application/json"   -d '{"name": "Jeremias"}'
```

which gives us the response

```bash
{"message": "Hello Jeremias!"}
```

> 💡 **TODO: HTTP vs REST API**
## Step 4: Add some security

By default, when you create an API Gateway endpoint, AWS gives you a **public URL**. Anyone who knows that URL could send requests. That’s fine for testing, but for a production setup we want **some form of security**.

|**Auth Method**|**How It Works**|**Best For**|**Secure?**|**Customizable?**|
|---|---|---|---|---|
|🔓 Open (No Auth)|No checks at all|Testing, public info|❌ No|❌ No|
|🔑 API Key|Static key in `x-api-key` header|Rate limiting, controlled public access|⚠️ Low|❌ No|
|🔐 IAM (SigV4 Signing)|Signed requests using AWS access keys + hashed signature|Internal/back-end service-to-service calls|✅ Yes|✅ IAM policies|
|👤 Cognito User Pools|JWT-based user login (Cognito handles identity)|Frontend/mobile apps with user login|✅ Yes|⚠️ Limited|
|🧠 Lambda Authorizer|Custom logic via Lambda to validate token or headers|Custom tokens (e.g. Auth0, Firebase, etc.)|✅ Yes|✅ Fully|
For our simple use case, API Key security is secure enough.
### **Step 4.1: Create an API Key**
1. From the API Gateway portal, in the left menu, click **API Keys → Create API Key**.
2. Give it a **name** like `VercelFrontendKey`.
3. Leave it as **Auto-generate** for the key value, or provide your own.
4. Click **Save**.

![[Pasted image 20251005202543.png]]
> An API key is a unique string (long, random-looking) that a client sends with a request to identify itself to the API - A sort of passcode.

### **Step 4.2: Create a Usage Plan**

1. In API Gateway console → **Usage Plans → Create**. A usage plan defines some limits to the request number, etc.
2. Give it a **name** like `PersonalProjectUsagePlan`.
3. Set **throttle** (rate limit: max requests per second ; burst limit: max requests that can be handled instantly) and **quota** (total requests per day/week/month) - I chose 10; 5 ; 100/day respectively. 
4. Click **Next**.
![[Pasted image 20251005203210.png]]

>  A usage plan is a way to associate API Keys with specific APIs, specifying how much traffic a key-holder is allowed to send. It's basically a ruleset for how someone can use the API. I
### **Step 4.3: Associate API Stages with Usage Plan**
1. On the **Associated Stages** page in the usage plan, click **Add API Stage**. A stage is essentially a deployment environment (like dev, test, prod). When we deploy an API, it must be deployed to a stage. The stage gets a unique URL e.g. `https://abc123.execute-api.us-east-1.amazonaws.com/prod`
2. Select your REST API (`egyptianArtAnalyzer`) and stage (`default`).
3. Click **Add**.

> Each time you deploy an API in API Gateway, you deploy it to a stage, which gives a  unique URL. Stages can have stage-specific settings (logging, throttling, variables) and provide a way to separate environments (dev, test, prod)
> When you create and deploy a new REST API for the first time, AWS will automatically create a stage called `default` - This one is just fine for our purposes.
> Stages can be used for Cannary Deployments, gradually rolling out new lambda versions and shifting traffic from one env to the other.

![[Pasted image 20251005204350.png]]
### **Step 4.4: Associate the API Key with the Usage Plan**
1. Go to **API Keys → select your key**.
2. Click **Add to Usage Plan** → select `PersonalProjectUsagePlan`.
3. Save.
![[Pasted image 20251005204931.png]]
![[Pasted image 20251005205003.png]]
### **Step 4.5: Require API Key in the Method**
1. Go to **API Gateway → API: YourAPI -> Resources → your POST method**.
2. Click **Method Request**.
3. Toggle **API Key Required → true**.
4. Deploy your API (Actions → Deploy API → choose `default` stage).

![[Pasted image 20251005205610.png]]

Test: only requests with the correct API key should succeed

```
curl -X POST https://XXXXXXX.execute-api.us-east-2.amazonaws.com/default/egyptianArtAnalyzer   -H "Content-Type: application/json" -H "x-api-key: YOUR-API-KEY" -d '{"name": "Jeremias"}'
{"message": "Hello Jeremias!"}
```

If you omit or provide the wrong key, you get:

```
curl -X POST https://XXXXXXX.execute-api.us-east-2.amazonaws.com/default/egyptianArtAnalyzer   -H "Content-Type: application/json"   -d '{"name": "Jeremias"}'
{"message":"Forbidden"}
```

The API key works basically like a password for your API.
* You should **never embed it in client-side code because anyone could inspect it and use it.
* Treat it like a **secret**: store it in environment variables on your server or Vercel serverless function.
### Recap

We added different pieces in the puzzle (Lambda Function, API, Usage Plan, API Key) and tied them together.

![[Pasted image 20251005211803.png]]
## Step 5: Running Lambda from a Docker Container 

Let's replace the inline hello world by custom code on a docker container. 

> When to use Docker? Normally, Lambda is deployed by uploading a ZIP file of your code. However, when we want more complex dependencies (e.g. use sth like Pillow that may require system libraries), we can package the code as a Docker image, upload it to ECR and let Lambda pull it. 
### Step 5.1 - Create a new Python repository with our custom lambda implementation

First, we need to write the Python code that we expect to run in Lambda. The following are simplified Python versions intended to illustrate the key pieces as pseudocode. The full source code can be found here.
#### Write the API

`lambda_function.py`  (simplified here) would be responsible from processing and validating HTTP POST request, returning an error if the request is malformed, forwarding the request to the `gemini_strategy` module otherwise, and finally returning successful results via HTTP.
 
```python
#... other imports ...
from gemini_strategy import analyze_egyptian_art_with_gemini

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
	"""
	AWS Lambda handler for Egyptian art analysis API
	"""
	
	# Only handle POST requests
	if event.get('httpMethod') != 'POST':
		# Return 405 error 
	
	# Parse the request body
	body = event.get('body', '')  # if body is empty, return error 400
	request_data = json.loads(body) # if json parsing fails, return error 400
	image_data = request_data.get('image') # if image is not set, error 400
	base64.b64decode(image_data)
	print(f"Received image: {len(image_data)} characters (base64)")
	gemini_result = analyze_egyptian_art_with_gemini(image_data, ...)
	
	# If Gemini analysis failed, return error 500
	# Else, return a status 200 (success) and the response
	
	return {
		'statusCode': 200,
		'headers': {
			'Access-Control-Allow-Origin': '*',
			'Content-Type': 'application/json'
		},
		'body': json.dumps(gemini_result, indent=2)
	}
```
#### Define the model

`gemini_strategy.py` is the actual logic of our lambda, where the ML magic takes place. Here we dynamically craft a prompt based on the request parameters, pass on the image into the prompt and request a structured output reply from Gemini.

```python
import google.generativeai as genai

def analyze_egyptian_art_with_gemini(image_data, speed='fast', image_type='unknown', thinking_budget=DEFAULT_THINKING_BUDGET):
	api_key = os.environ.get('GOOGLE_API_KEY')
	genai.configure(api_key=api_key)
	image_bytes = base64.b64decode(image_data)
	image = PIL.Image.open(io.BytesIO(image_bytes))
	prompt_text = create_egyptian_art_prompt(image_type)
	model_name = speed_to_model.get(speed, 'gemini-2.5-flash')
	
	
	model = genai.GenerativeModel(
		model_name=model_name,
		generation_config=genai.types.GenerationConfig(
			response_schema=EgyptianArtAnalysis,
			response_mime_type="application/json",
			temperature=0
		)
	)
	
	response = model.generate_content([prompt_text, image])
	
	# (...)
	
	return formatted_response
```

### Step 5.2 - Deployment infrastructure

#### 5.2.0 - Delete the old (zip) function

AWS Lambda has two different deployment methods:
1. ZIP Package (current setting of our function created via the API):
	* Upload a ZIP file with your code
	* AWS runs it using their runtime
	* Limited to specific runtimes (Python 3.12, Node.js, etc.)
2. Docker Container (desired new method):
	* Upload a Docker image
	* AWS runs your container
	* You control the entire environment
	* Can use any runtime, any dependencies

AWS does NOT allow you to change the package type of an existing function, so we'll just get rid of the ZIP version and replace it by a Docker version.

```bash
$ aws lambda get-function --profile egyptian-project --region us-east-2 --function-name egyptianArtAnalyzer

"Configuration": {
	"FunctionName": "egyptianArtAnalyzer",
	(...)
	"PackageType": "Zip",  # THIS IS INCOMPATIBLE WITH A DOCKER DEPLOYMENT
```

```bash
$ aws lambda delete-function --profile egyptian-project --region us-east-2 --function-name egyptianArtAnalyzer
```

When we create a new Lambda with the same name, the API will automatically reconnect with it and our settings will be preserved.

```bash
export GOOGLE_API_KEY=XXXXXXXXXXXXXXXX # Put your Gemini key here
&& chmod +x deploy.sh && ./deploy.sh
```

#### 5.2.1 - Write a Dockerfile

Setup a docker container by defining a `Dockerfile`:

```dockerfile
FROM public.ecr.aws/lambda/python:3.11

# Copy requirements file
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install dependencies
RUN pip install -r requirements.txt

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY gemini_strategy.py ${LAMBDA_TASK_ROOT}
COPY schemas.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD [ "lambda_function.lambda_handler" ]
```

* AWS Lambda Python 3.11 is the base image
* Dependencies are installed from `requirements.txt`, which is a simple text file as follows:
```
google-generativeai==0.8.5
Pillow==10.4.0
pydantic==2.10.4
```
* Copy all python modules and set the Lambda handler

#### 5.2.2 - Write a deployment script

Let's write a handy deployment script that we can reuse every time we need to redeploy our code, `deploy.sh`.

To start with, we must fill in the appropriate constants at the top:

```bash
FUNCTION_NAME="egyptianArtAnalyzer"
REGION="us-east-2"
PROFILE="egyptian-project"
ACCOUNT_ID=$(aws sts get-caller-identity --profile ${PROFILE} --query Account --output text)
ECR_REPOSITORY="jere-egyptian-art-analyzer" 
IMAGE_TAG="latest"
```

**Note**: Function name must match the lambda name ; profile must match the profile name you set in step 1

Then we build the docker image, following the instructions on the docker file:

```bash
echo "Building Docker image..."
sudo docker build -t ${ECR_REPOSITORY}:${IMAGE_TAG} .
```

Next, we create an ECR repository if it does not exist.

```bash
aws ecr create-repository --profile ${PROFILE} --repository-name ${ECR_REPOSITORY} --region ${REGION} 2>/dev/null || echo "Repository already exists"
```

> **ECR (Elastic Container Registry)** is a fully-managed **Docker container registry** by AWS — kind of like **Docker Hub**, but private (or optionally public) and integrated with AWS services like **Lambda**, **ECS**, and **EKS**.
> In case you don't know, Docker Hub is like the app store for Docker images - A public place where developers upload and share these packages (images) . There you can find official images like python:3.11 or node:18.
> **ECR** is AWS's own version of Docker Hub, but private by default and  integrated with AWS services.  In this context, we are going to use it to store out Docker image (the packaged Lambda code) and then deploy that image.

```bash
echo "Creating ECR repository if it doesn't exist..."
aws ecr create-repository --profile ${PROFILE} --repository-name ${ECR_REPOSITORY} --region ${REGION} 2>/dev/null || echo "Repository already exists"

echo "Logging in to ECR..."
aws ecr get-login-password --profile ${PROFILE} --region ${REGION} | sudo docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

echo "Tagging image for ECR..."
sudo docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}

echo "Pushing image to ECR..."
sudo docker push ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}
```

About tagging: 
- Docker needs the image to have the **right destination tag** (i.e., where you're pushing it)
- This command **retags your local image** with the **ECR repository address**, which looks like: `123456789012.dkr.ecr.us-east-1.amazonaws.com/my-repo:latest
`
After this, AWS can pull this image. Let's follow a classic pattern: "try to create; if it already exists update instead"

```bash
echo "Creating or updating Lambda function..."
aws lambda create-function \
    --profile ${PROFILE} \
    --function-name ${FUNCTION_NAME} \
    --package-type Image \
    --code ImageUri=${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG} \
    --timeout 30 \
    --memory-size 512 \
    --region ${REGION} 2>/dev/null || \
aws lambda update-function-code \
    --profile ${PROFILE} \
    --function-name ${FUNCTION_NAME} \
    --image-uri ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG} \
    --region ${REGION}
```

The function should already exist given that we created it manually before. Notes:
- `--timeout 30`: Sets the **maximum amount of time (in seconds)** your Lambda function is allowed to run.
- `--memory-size 512`:  Allocates **512 MB of memory** to your Lambda function (and CPU is allocated proportionally)

Last but not least, we set the Gemini API key:

```bash
echo "Setting environment variables..."
aws lambda update-function-configuration \
    --profile ${PROFILE} \
    --function-name ${FUNCTION_NAME} \
    --environment Variables="{GOOGLE_API_KEY=${GOOGLE_API_KEY}}" \
    --region ${REGION}
```

Finally:

```bash
echo "Deployment complete!"
echo "Function ARN: arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}"
```

## Step 6: How steps 2 to 5 could've been done via `aws-cli` 

Note: The whole process of setting up the lambda could've been made using aws-cli instead of going through the GUI - I used the GUI to introduce new concepts:

```bash
# === Configuration ===
export PROFILE="egyptian-project"
export REGION="us-east-2"
export FUNCTION_NAME="egyptianArtAnalyzer"
export ECR_REPOSITORY="jere-egyptian-art-analyzer"
export IMAGE_TAG="latest"
export API_NAME="egyptianArtAPI"
export USAGE_PLAN="PersonalProjectUsagePlan"
export API_KEY_NAME="VercelFrontendKey"
export STAGE="default"

# Get your AWS account ID automatically
export ACCOUNT_ID=$(aws sts get-caller-identity --profile $PROFILE --query Account --output text)

# Create ECR repository
aws ecr create-repository \
  --repository-name $ECR_REPOSITORY \
  --region $REGION \
  --profile $PROFILE 2>/dev/null || echo "Repository already exists"

# Authenticate docker to ECR and tag/push the image
aws ecr get-login-password --region $REGION --profile $PROFILE | \
sudo docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Build Docker image
sudo docker build -t ${ECR_REPOSITORY}:${IMAGE_TAG} .

# Tag it for ECR
sudo docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}

# Push it
sudo docker push ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}

# Create the lambda execution role
aws iam create-role \
  --role-name lambda-execution-role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Principal": {"Service": "lambda.amazonaws.com"},
        "Action": "sts:AssumeRole"
      }
    ]
  }' \
  --profile $PROFILE

# Attach the AWS-managed policy for logging
aws iam attach-role-policy \
  --role-name lambda-execution-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole \
  --profile $PROFILE

# Create the Lambda function w/ docker image
aws lambda create-function \
  --function-name $FUNCTION_NAME \
  --package-type Image \
  --code ImageUri=${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG} \
  --role arn:aws:iam::${ACCOUNT_ID}:role/lambda-execution-role \
  --region $REGION \
  --timeout 30 \
  --memory-size 512 \
  --profile $PROFILE

# Add Gemini API key as env variable
aws lambda update-function-configuration \
  --function-name $FUNCTION_NAME \
  --environment Variables="{GOOGLE_API_KEY=YOUR_GEMINI_API_KEY}" \
  --region $REGION \
  --profile $PROFILE

# Create a REST API in API Gateway
API_ID=$(aws apigateway create-rest-api \
  --name "$API_NAME" \
  --region $REGION \
  --profile $PROFILE \
  --query 'id' \
  --output text)

echo "API ID: $API_ID"

ROOT_ID=$(aws apigateway get-resources \
  --rest-api-id $API_ID \
  --region $REGION \
  --profile $PROFILE \
  --query 'items[0].id' \
  --output text)

echo "Root ID: $ROOT_ID"

# Add the POST method
aws apigateway put-method \
  --rest-api-id $API_ID \
  --resource-id $ROOT_ID \
  --http-method POST \
  --authorization-type "NONE" \
  --api-key-required true \
  --region $REGION \
  --profile $PROFILE
  
#Link the POST method to your Lambda function
aws apigateway put-integration \
  --rest-api-id $API_ID \
  --resource-id $ROOT_ID \
  --http-method POST \
  --type AWS_PROXY \
  --integration-http-method POST \
  --uri arn:aws:apigateway:${REGION}:lambda:path/2015-03-31/functions/arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}/invocations \
  --region $REGION \
  --profile $PROFILE
  
#Give API Gateway permission to invoke the Lambda
aws lambda add-permission \
  --function-name $FUNCTION_NAME \
  --statement-id apigateway-access \
  --action lambda:InvokeFunction \
  --principal apigateway.amazonaws.com \
  --source-arn arn:aws:execute-api:${REGION}:${ACCOUNT_ID}:${API_ID}/*/POST/ \
  --region $REGION \
  --profile $PROFILE

# Deploy the API
aws apigateway create-deployment \
  --rest-api-id $API_ID \
  --stage-name $STAGE \
  --region $REGION \
  --profile $PROFILE
  
# Create API Key and Usage plan
API_KEY_ID=$(aws apigateway create-api-key \
  --name "$API_KEY_NAME" \
  --enabled \
  --region $REGION \
  --profile $PROFILE \
  --query 'id' \
  --output text)

API_KEY_VALUE=$(aws apigateway get-api-key \
  --api-key $API_KEY_ID \
  --include-value \
  --region $REGION \
  --profile $PROFILE \
  --query 'value' \
  --output text)

echo "API Key created: $API_KEY_VALUE"
USAGE_PLAN_ID=$(aws apigateway create-usage-plan \
  --name "$USAGE_PLAN" \
  --throttle rateLimit=10,burstLimit=5 \
  --quota limit=100,period=DAY \
  --region $REGION \
  --profile $PROFILE \
  --query 'id' \
  --output text)

echo "Usage Plan ID: $USAGE_PLAN_ID"

#Link API stage to usage plan
aws apigateway create-usage-plan-key \
  --usage-plan-id $USAGE_PLAN_ID \
  --key-id $API_KEY_ID \
  --key-type "API_KEY" \
  --region $REGION \
  --profile $PROFILE

aws apigateway create-usage-plan-key \
  --usage-plan-id $USAGE_PLAN_ID \
  --key-id $API_KEY_ID \
  --key-type API_KEY \
  --region $REGION \
  --profile $PROFILE 2>/dev/null || echo "API key already added"
# Link API+stage with usage plan
aws apigateway update-usage-plan \
  --usage-plan-id $USAGE_PLAN_ID \
  --patch-operations op=add,path=/apiStages,value="${API_ID}:${STAGE}" \
  --region $REGION \
  --profile $PROFILE

# Test
curl -X POST \
  "https://${API_ID}.execute-api.${REGION}.amazonaws.com/${STAGE}/" \
  -H "Content-Type: application/json" \
  -H "x-api-key: ${API_KEY_VALUE}" \
  -d '{"name": "Jeremias"}'
```

Recap:

| Component                | Created By                                           | Description                            |
| ------------------------ | ---------------------------------------------------- | -------------------------------------- |
| **ECR**                  | `aws ecr create-repository`                          | Stores your Docker image               |
| **Lambda**               | `aws lambda create-function`                         | Runs the model inside the Docker image |
| **IAM Role**             | `aws iam create-role`                                | Gives Lambda permissions to log        |
| **API Gateway**          | `aws apigateway create-rest-api`                     | Exposes Lambda via HTTP                |
| **API Key + Usage Plan** | `aws apigateway create-api-key`, `create-usage-plan` | Secures and throttles API              |
| **Stage**                | `aws apigateway create-deployment`                   | Deploys API version                    |
| **Test**                 | `curl`                                               | Invokes the endpoint with a valid key  |
## Step 7 - How to invoke the lambda from client code

### What are CORS?

A **CORS request** (Cross-Origin Resource Sharing request) is an HTTP request made by a web page (usually JavaScript in a browser) to a **different origin** — meaning a different **domain, protocol, or port** — than the one that served the web page.

E.g. `https://mywebsite.com` makes a request to `https://api.othersite.com/data`

In my case, I created my own blogpost (hosted in Vercel, [jeremias-rodriguez.com](https://www.jeremias-rodriguez.com/)), and I would like my web page to make a request to my AWS lambda (on the amazonaws.com domain). This is, therefore, a CORS request.

Therefore, every AWS lambda function needs CORS configured if it's called from a browser. AWS philosophy is "secure by default", so they don't assume we may want to allow cross-origin requests.
### Handling CORS
In order to allow CORs, it is necessary to add the OPTION method to handle preflight request. One way to do it is:

```
# Create a new method (OPTIONS) for your resource in API Gateway
aws apigateway put-method --http-method OPTIONS

# Tell API Gateway what kind of response to return for that method (e.g., a 200 OK)
aws apigateway put-method-response --status-code 200

# Configure a **MOCK integration**, meaning API Gateway responds directly (without invoking Lambda). This is common for CORS preflight, since it doesn’t need to trigger your Lambda
aws apigateway put-integration --type MOCK

# Specifie what headers to include in the response — for CORS, this should include: Access-Control-Allow-Origin: '*' ; Access-Control-Allow-Methods: 'GET,POST,OPTIONS' ; Access-Control-Allow-Headers: 'Content-Type'
aws apigateway put-integration-response

# Deploy the updated API to the default stage
aws apigateway create-deployment --stage-name default
```

Why do we do this? When a browser like Google Chrome tries to make a CORS request (e.g. a POST sending an image to run our image analysis model), it sends an OPTIONS request to API Gateway. If API Gateway does not support OPTIONS method, then the POST request will not be made.

The commands define a dummy `OPTIONS` endpoint that **replies instantly** with CORS headers, satisfying the browser’s preflight check.


Once we support OPTIONS method, the browser will do as follows:

1. Send an OPTIONS request
		```
		OPTIONS /myendpoint
		Origin: https://www.jeremias-rodriguez.com
		Access-Control-Request-Method: POST
		Access-Control-Request-Headers: Content-Type
		```
2. Receive a successful status response (200)
		```
		HTTP/1.1 200 OK
		Access-Control-Allow-Origin: https://www.jeremias-rodriguez.com
		Access-Control-Allow-Methods: GET, POST, OPTIONS
		Access-Control-Allow-Headers: Content-Type
		Content-Length: 0
		```
3. Send the POST request
4. Receive the ML model result

This could've also been done from the GUI:

![[Pasted image 20251020000829.png]]

### Client Code

Example: How to call this API from a website hosted in Vercel and implemented in typescript.

```typescript
  const response = await fetch(lambdaUrl, {
	method: "POST",
	headers: {
	  "Content-Type": "application/json",
	  "x-api-key": apiKey,
	},
	body: JSON.stringify({
	  image: imageBase64,
	  speed: speed,
	  imageType: imageType,
	}),
  })
```

Secrets:
- API URL: Stored in NEXT_PUBLIC_LAMBDA_API_URL environment variable
- API Key: Stored in NEXT_PUBLIC_API_KEY environment variable
- Location: .env.local file (local development) and Vercel environment variables (production)

This setup leaves the lambda API url and API key accessible to anyone if they look for it. Therefore, we rely on usage limits for the API key, and we'll update the CORS config so that only requests from the domain jeremiasrodriguez.com are allowed.

NEXT STEPS:
- CORS make only domain reqs allowed
- Organize the whole post again
- improve data science part, new repo!
- write monitoring
- up next: custom model

## AWS Lambda Monitoring

# Gemini Monitoring
https://aistudio.google.com/u/0/usage

# MFA

# Budget

* Set up a budget alarm
	- Login as root user - Go to **Billing and Cost Management > Budgets**.
	- Create a budget (say $10/month).	    
	- Add an alert to email you if you exceed $5 or $10.

![[Pasted image 20250921104302.png]]
###  Monitoring

gemini side: 
aws: document

# FAQs:
* Recap: Gateway API, HTTP
* Roles vs Users:

| Feature                    | IAM User                       | IAM Role                                 |
| -------------------------- | ------------------------------ | ---------------------------------------- |
| Has long-term credentials? | ✅ Yes (password / access keys) | ❌ No, temporary credentials only         |
| Meant for                  | Humans or apps                 | AWS services or cross-account access     |
| Example use                | You using CLI                  | Lambda reading from S3, EC2 accessing S3 |
| Assigned to                | Person/app                     | Service (Lambda, EC2, etc.)              |
|                            |                                |                                          |
# ECR

* **Amazon Elastic Container Registry (ECR)** is AWS’s **fully-managed Docker container registry**. Think of it like Docker Hub, but hosted on AWS and integrated with Lambda, ECS, and Fargate.
	* - You **push your Docker images** (your Lambda packaged as a container) to ECR.
	- Then, Lambda can **pull the image directly** from ECR and run it.
    
- It’s private by default, so only your AWS account (or other accounts you allow) can access the images.