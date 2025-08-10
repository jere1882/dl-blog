---
tags:
  - Computer
  - Vision
aliases: 
publish: false
slug: mlops notes
title: Notes on MLOps
description: Assorted notes and takeaways
date: 2024-11-26
image: /thumbnails/backbones.png
---

# REST APIs

## What is a REST API?
REST (REpresentational State Transfer) APIs are a foundational technology for deploying and accessing ML models in production environments. More broadly, it is the most common communication standard for computers over the internet.

A REST API is a way to interact with a server using standard web protocols. It's a set of rules and good practices, not a specific library or toolkit. It asserts principles such as:

* client - server
* **Statelessness**: Each request from a client contains all the information needed for the server to process it. C and S don't need to store any information about each other.
* **Uniform interface**: Uses standard HTTP methods (GET, POST, PUT and DELETE)
* Data is treated as a **resource** that can be retrieved or manipulated
* **cacheable**

API: Application Programming Interface

```
       request
CLIENT ----------->  SERVER
       <----------
		response
```
An API that adhers to REST is a RESTFul API.

## REST API components

**Resource**: A single piece of data or an abstraction of a data entity that the API provides access to. E.g. user, product, image, model predictions.

**URI**: Uniform Resource Identifier is the specific address used to locate a resource on the web. E.g. a URI pointing to the collection of all users might be https://api.example.com/users

**Endpoint**: A specific URL combined with an HTTP method that defines the action you can perform on a resource. It is the actionable point of interaction with the API.
* Each endpoint corresponds to an operation (GET, POST, etc)
* Endpoint := a URL where the API is accessible e.g. https://api.example.com/predict

The CLIENT makes a REQUEST  to the endpoint for the resource over HTTP. An HTTP request has a specific method (verb) associated:

* **GET**: Retrieve info e.g. model metadata
* **POST**: Send data e.g. send input data for a prediction as json or image (the request body)
* **DELETE**: remove a resource (not used much for ML)
* **PUT**: Update a resource (not used much for ML)

The request has five fields:
1. method
2. request target URI
3. HTTP version e.g. HTTP/1.1 or HTTP/2
4. Header: metadata about the request 
5. Body optional e.g. a JSON XML or even an image.

The SERVER replies with an HTTP status code:
* 200 - success
* 400 - failure, bad request
* 500 - failure, server error

As well as:
1. HTTP version
2. headers
3. body, optionally

HTTP is an **abstraction layer** built on top of TCP (and more recently, UDP in HTTP/3) to simplify communication over the internet. It allows you to focus on high-level concepts like requests and responses without worrying about the low-level details of how data is transmitted.

## Libraries and frameworks to implement REST

**Flask:** Lightweight and flexible python web framework, for simple REST APIs
* Pros: **Mature**, widely used, **simple**, flexible, lightweight
* Cons: No native **async** support, manual input validation, less performant

**FastAPI:** Modern Python framework, designed specifically for creating APIs quickly and efficiently.
* Pros: **Fast** performance, optimized for **concurrency**, input **validation** using type hints
* Cons: Less mature (2018), overhead for small projects.

These frameworks make it easy to:
* **Define what happens when you get an HTTP request (GET POST PUT DELETE)**
* **Data serialization e.g. to-from json**
* Routing: **define endpoints** like `/predict`

## Creation of a RESTful API,high level overview
1. Train a model
2. Create an API server using a web framework like FLASK or FASTAPI
3. Define endpoints e.g.

```
\predict -> Making predictions
\health -> Check if server is running
\metadata -> provide model details 
```

4. Deploy the server: Host the API On a cloud server or a server (e.g. AWS, GCP)

## Example with Flask

Let's build a REST API using Python and Flask
```python

from flask import Flask, request, jsonify
import joblib

# Load your trained model
model = joblib.load("my_model.pkl")

app = Flask(__name__)

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running"})

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get JSON data from request body
        features = data['features']  # Extract features
        prediction = model.predict([features])  # Make prediction
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

```
Note: `__name__` is a special built in variable in Python, the name of the current module. If the script is being run directly, then `'__name__' == '__main__'`. If the script is just being imported, then they are not equal and we don't run all the setup.

We can test it using `curl`:

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" \
-d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

or we can test it with python:

```python
import requests

url = "http://localhost:5000/predict"
data = {"features": [5.1, 3.5, 1.4, 0.2]}
response = requests.post(url, json=data)
print(response.json())
```

## Example with Flask: Image data

If this is e.g. an object detection model, we can send images via POST request to the REST API.

We send it like this:

```python
import requests

# Specify the endpoint URL
url = "http://your-api-url.com/predict"

# Open the image file
with open("input_image.jpg", "rb") as file:
    response = requests.post(url, files={"file": file})

# Get the response
print(response.json())
```

The server handles it:

```python
from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    # Load the image
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
    
    # Process the image and run the model (example only)
    result = {"detections": [{"class": "cat", "confidence": 0.95}]}

    # Flask automatically assigns a 200 status to responses if no specific status code is set     
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

```

Client code in a website or mobile app can interact with the REST API by sending HTTP requests e.g:
* JavaScript's `fetch()` or `axios`
* Python's `request` library

## Going from this toy example to reality

1. Train and save the model e.g. a .pt or .pkl format ;  optimize it for efficiency e.g. ONNX
2.  Develop the API e.g. Fast/FastAPI 
3. Containarize the application by creating a Dockerfile to package API+Model

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```
4. Deploy to cloud.

So we have Model as ONNX ; `app.py` API ; `dockerfile`. How do we go on? Deployment directly to AWS EC2

- Login to AWS Management Constole
- Navigate to EC2 and launch a new instance (ubuntu 20.04 ; select computing power)
- Launch the instance and note its public address

- ssh into the instance `ssh -i your-key.pem ubuntu@<your-ec2-public-ip>` and sudo install docker
- scp the dockerfile, flask app and model to the instance
- build and run the docker image `sudo docker build -t flask-api .`
- run the container (mapping ports) `sudo docker run -d -p 80:5000 flask-api`

Now you can access the API e.g.
`curl -X POST -F "file=@input_image.jpg" http://<your-ec2-public-ip>/predict`

Note: `build` the dockerfile creates a (static) docker image by running the instructions, it's just the blueprint. `run` the docker container will start an instance of the docker image where the application will run, isolated.

This example deploys directly in AWS no kubernetes in between.

ALTERNATIVES TO AWS

You must pick a hosting platform, but the choice depends on our needs. Cloud services such as AWS / GCP / Azure are widely used in professional environemtns. E.g. a flask API behing AWS Elastic Load Balancer.

## ADDING KUBERNETS: WHEN TO DO IT?

Containerized deployment (using Docker and optionally Kubernetes) is not strictly necessary but is becoming the industry norm. When to use kubernetes+AWS vs just AWS?

Kubernetes is an orchestration platform designed to manage multiple containers across many servers. It helps with:
- Scaling applications automatically based on demand.
- Load balancing across multiple instances of your application.
- Managing container lifecycle (e.g., restarting failed containers).
- Deploying rolling updates to your application.
- Running distributed systems efficiently.

You should consider Kubernetes when:
- High traffic or need to scale
- Require high availability
- Managing a cluster of servers
- you want cloud independence (works in any)

You can skip kubernetes and deploy directly to AWS when:
- Small development e.g. a flastk API for a small group of users on an EC2 instance
- If you can handle expected traffic with one or two server instances and don’t need automatic scaling, Kubernetes might be overkill. Deploying a small API for an ML model on AWS EC2 with a load balancer.

Recomendation: Start small with direct cloud deployment using Docker. Once your service grows and you need scalability or resilience, consider adding Kubernetes to manage the complexity.

## ADDING KUBERNETES: EXAMPLE

Summary of Tools and Technologies Involved:
- Docker: For containerizing the Flask API and ONNX model.
- AWS ECR: For storing your Docker image.
- AWS EKS: To manage your Kubernetes cluster.
- Kubernetes: For deployment and orchestration of your Flask API.
- LoadBalancer: To expose your service to the outside world.

Breakdown step by step:

### Step 1: Prepare a Flask API
```python
from flask import Flask, request, jsonify
import onnx
import numpy as np
import torch
import onnxruntime as ort

app = Flask(__name__)

# Load ONNX model (once at start-up)
model_path = "model.onnx"
onnx_model = ort.InferenceSession(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array(data["input"])  # Assuming input is a list or array
    ort_inputs = {onnx_model.get_inputs()[0].name: input_data}
    ort_outs = onnx_model.run(None, ort_inputs)
    prediction = ort_outs[0]
    return jsonify({"prediction": prediction.tolist()})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
```

### Step 2: Dockerize

Dockerfile looks like this:

```dockerfile
# Start with the base Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the necessary files
COPY app.py /app/
COPY model.onnx /app/

# Install dependencies
RUN pip install flask onnxruntime numpy

# Expose port for Flask API
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
```

### Step 3: Push Docker Image to a Container Registry

A container registry is a service or storage system used to store and manage Docker images (or container images). When you build a Docker image (which is a self-contained unit with your app and dependencies), you need a place to store it so that it can be accessed, downloaded, and run on any machine (e.g., on your production server or cloud instance).

A container registry serves as a central repository for Docker images, and it’s similar to version control systems like Git but for containerized applications.

Why?
- centralized storage
- versioning
- automated deployment
- scalability

Standard container registries:
- docker hub
- AWS Elastic Container Registry (ECR)
- Google Container Registry (GCR) ; Azure container ergistru

These container registries provide both public and private repositories where you can store and access your container images.

Do not confuse with a MODEL REGISTRY. In the ML model deployment context, a model registry could be used instead of or in addition to a container registry. . A model registry keeps track of different versions of models, allows for easy access and retrieval of models, and supports metadata such as performance metrics, model lineage, and deployment history. 
- Manage model versions
- Monitor models in production
Technologies:
- MLFlow
- Kubeflow

You may use both, a model registry and a container registry.
Anyways, to push the docker container to a container registry you build it:

```
docker build -t my-flask-api .
```
then you login to AWS ECR, create a repository and push it

```
aws ecr create-repository --repository-name my-flask-api
docker tag my-flask-api:latest <aws-account-id>.dkr.ecr.<your-region>.amazonaws.com/my-flask-api:latest
docker push <aws-account-id>.dkr.ecr.<your-region>.amazonaws.com/my-flask-api:latest
```
### Step 4: Set Up AWS EKS (Elastic Kubernets Service)
We can create a EKS cluster (AWS Managed kubernets service) using AWS CLI:

```
aws eks create-cluster --name my-cluster --role-arn arn:aws:iam::<aws-account-id>:role/eks-service-role --resources-vpc-config subnetIds=<subnet-id>,securityGroupIds=<security-group-id>
```
And configure kubectl to interact with the EKS cluster

```
aws eks update-kubeconfig --name my-cluster --region <your-region>
```
### Step 5: Deploy your Flask API on Kubernetes

First we define deployment.yaml, specifying how the API should be deployed:

```yaml
apiVersion: apps/v1
kind: Deployment # Indicates that this YAML file defines a Deployment resource.
metadata:
  name: flask-api
spec:
  replicas: 2  # Number of replicas (pods) for your app
  selector:
    matchLabels:
      app: flask-api
  template:
    metadata:
      labels:
        app: flask-api
    spec:
      containers:
        - name: flask-api
          image: <aws-account-id>.dkr.ecr.<your-region>.amazonaws.com/my-flask-api:latest
          ports:
            - containerPort: 5000 #Specifies that the Flask app listens on port 5000 inside the container.
```

The service.yaml file defines the Service resource, which is responsible for exposing your Flask API to the outside world, or to other services within the Kubernetes cluster. The service will route traffic to the Flask API pods, enabling users or other services to interact with the API.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: flask-api-service
spec:
  selector:
    app: flask-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
  type: LoadBalancer  # To expose your app to the internet
```

spec.type: LoadBalancer: This specifies that the service should be exposed externally through a LoadBalancer. A LoadBalancer is a cloud-based resource that distributes incoming network traffic across multiple pods. It gives you a public IP address or DNS name so that external clients can access your Flask API.

Once the Kubernetes deployment and service configurations are defined, you need to apply them to your Kubernetes cluster using the kubectl command-line tool.

This command will create or update the Deployment resource in Kubernetes based on the deployment.yaml configuration. It will ensure that two replicas of the Flask API are running (as defined in the replicas: 2 section), and that they are using the Docker image you specified:

```bash
kubectl apply -f deployment.yaml
```

This command will create or update the Service resource, exposing your Flask API to the internet (or within your Kubernetes cluster) based on the service.yaml configuration. Since the service is of type LoadBalancer, a cloud load balancer will be provisioned to expose your Flask API:

```bash
kubectl apply -f service.yaml
```
Check status:

```bash
kubectl get pods  # Check if the pods are running
kubectl get services  # Get the external IP (LoadBalancer URL)
```
example output of the alst command:

```bash
NAME               TYPE           CLUSTER-IP      EXTERNAL-IP     PORT(S)        AGE
flask-api-service  LoadBalancer   10.100.200.10   <external-ip>   80:30323/TCP   10m
```

EXTERNAL-IP: This is the public IP or DNS address you can use to access the Flask API from the internet. Once it's assigned (it may take a minute or two), you can send requests to this IP on port 80.

### Step 6: Expose the API

After deploying the service, Kubernetes will provision an external IP or LoadBalancer for your service.

So, if the LoadBalancer provisioned by Kubernetes provides an external IP like `http://<external-ip>`, clients will send requests to `http://<external-ip>:80`. Kubernetes will internally forward the requests to port 5000, where your Flask app is listening.

####  Autoscaling

To automatically scale the number of Pods running your Flask API based on demand (e.g., CPU usage or request rate), you can use Horizontal Pod Autoscaler (HPA).

You define a `hpa.yaml` defining things like

```yaml
  minReplicas: 2  # Minimum number of replicas
  maxReplicas: 10  # Maximum number of replicas
  averageUtilization: 50  # Scale if CPU usage is above 50%
```

And then apply it

```bash
kubectl apply -f hpa.yaml
```
### Caching
Caching can be implemented in several layers depending on the needs of your application. Here are some options:
1. Flask API level caching: in-memory caching using libraries like Flash Cachhing

```python
from flask_caching import Cache
app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/predict', methods=['POST'])
@cache.cached(timeout=60)  # Cache the result for 60 seconds
def predict():
    # your prediction logic
    return jsonify(result)
```
2. kubernetes level kaching
Kubernetes itself does not provide built-in caching, but you can deploy caching tools like Redis or Memcached alongside your application. These tools can be used to cache responses or other data externally, reducing the load on your Flask API and improving performance.

Redis or Memcached can be deployed as separate Pods in your Kubernetes cluster, and your Flask API can interact with them to cache responses.

3. cloud level caching
If you're using a cloud platform like AWS, caching might be offloaded to managed services:

4. CDN for static content

## Concurrency and Gunicorn

By default flask has a queue, If two requests are sent simultaneously, Flask will handle them sequentially. For production environments, the built-in Flask server is not recommended because it isn't optimized for concurrent processing. Instead, you deploy Flask with a production-grade web server like Gunicorn or uWSGI. Handle multiple requests simultaneously by using multiple worker processes or threads. 

You can run your Flask app with Gunicorn and specify the number of worker processes:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```
-w 4: Launches 4 worker processes to handle requests.

Each worker can handle one request at a time. Multiple workers enable the server to process multiple requests in parallel.

If your application is overwhelmed by requests, you can scale horizontally by running multiple containers. To manage high traffic, you can deploy multiple instances of your Flask API (containers) behind a load balancer.
## FastAPI Demo

```python
from fastapi import FastAPI
from pydantic import BaseModel
from aiocache import Cache
from aiocache.serializers import JsonSerializer
import time

# Initialize FastAPI app
app = FastAPI()

# Initialize caching (in-memory cache)
cache = Cache.from_url("redis://localhost:6379", serializer=JsonSerializer())

# Example input data model
class PredictionRequest(BaseModel):
    feature1: float
    feature2: float

# Simulate a model prediction function (you can replace this with your actual model logic)
def model_predict(feature1: float, feature2: float):
    # Dummy prediction logic
    time.sleep(2)  # Simulate a delay (e.g., processing time for ML model)
    return {"result": feature1 * feature2}

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Cache key based on input features
    cache_key = f"predict_{request.feature1}_{request.feature2}"
    
    # Check if the result is already cached
    cached_result = await cache.get(cache_key)
    if cached_result:
        return cached_result  # Return cached result if available
    
    # If no cached result, make the prediction
    result = model_predict(request.feature1, request.feature2)
    
    # Cache the result for 60 seconds
    await cache.set(cache_key, result, ttl=60)
    
    return result

```

- aiocache: This library is used to implement caching in an asynchronous way. In this example, it's using Redis as the backend store (redis://localhost:6379), but you can configure it to use other backends like Memcached or in-memory.

- Cache TTL: The ttl=60 option in await cache.set(...) ensures the result is cached for 60 seconds.

To run the app use uvicorn
`uvicorn main:app --reload`

## Uvicorn
In the context of APIs for ML, Uvicorn is an asynchronous web server used to run and serve machine learning models through APIs. When you build a REST or FastAPI-based service to expose a machine learning model, Uvicorn is the server that listens for incoming HTTP requests, handles them asynchronously, and sends back the prediction results.

## FAQ

### How do you scale a REST API for a model that receives thousands of request per second?
Use a load balancer, cache common predictions and deploy the model as microservices.

### Not to use REST

Issue: Some ML applications require persistent state across interactions. For example:
Chatbot systems: A conversation may depend on the context of previous interactions.
Iterative processes: Applications like Reinforcement Learning may need to store session-specific data (e.g., the state of an environment).
Alternative Technologies:
WebSockets: Enables bi-directional, real-time communication and is better suited for maintaining session context.
gRPC: Supports persistent streams for handling stateful operations efficiently.


### How would you handle model updates without downtime?
Use rolling updates or versioned APIs to deploy new models gradually

## Good practices
Validation: Validate input data to avoid runtime errors.
Logging: Log requests and responses for debugging.
Versioning: Use versioned endpoints (e.g., /v1/predict) for backward compatibility.
Scalability: Use tools like Kubernetes or load balancers to handle high traffic.
Security: Implement authentication (e.g., API keys, OAuth) and use HTTPS.
caching

## What is your stack question.

Linux, C++

Data pipeline and preprocessing:
- Python
- Pandas, numpy. Trying to get good at Apache Spark (I know it is very useful Spark+)
- h5 for large data??
- OpenCV

Data science:
- Python (matplotlib, seaborn)
- SQL
- Mode 

Model development:
- scikit learn for traditiona ML ; several years ago (about 5 years ago) I had the change to go over much of the interface
- pytorch  ; pytorch ligning
- pretrained models from OpenMMLab ; hugging face
- I have done tutorials on keras, tensorflow, etc. but no professional experience on that.
- ONNX for optimizaton
- w&b 

Deployment:
- Docker containerization
- FastAPI / Flask
- Embedded devices, no cloud.
- Cloud deployemnt. kubernetes, AWS EKS / ECS.

Jupyter. VScode, especially remotely.

Git. Github. Bitbucket. 

# AWS Lambda deployment and API Gateway (no Flask  / FastAPI)

There are several AWS alternatives to deploying and running ML models.
![Pasted image 20241220125805](/assets/Pasted%20image%2020241220125805.png)
**AWS Lambda** is a **serverless compute service** that runs your code in response to events without requiring you to provision or manage servers. You upload your code, and Lambda handles the execution, scaling, and management.

- Fast inference for small model
- You only pay when the model is actually invoked, cost effective for low traffic
- Lambda can trigger model inference in response to events (e.g., API Gateway requests, S3 uploads, or database updates).

Example:

- Export the trained model e.g. ONNX or Pytorch
- Bundle the model with a lighweight runtime (e.g. Python and dependencies) on a container
- Upload the packaged model and inference logic to AWS Lambda
- Set triggers like a HTTP API
- The Lambda function loads the model and processes incoming requests.

If you want your model to be accessible via HTTP (e.g., to support external apps or front-end services), you can combine Lambda with **API Gateway** to create a RESTful API.

**API Gateway** is a fully managed service provided by cloud platforms like AWS, Azure, and GCP that enables you to create, publish, secure, and manage APIs without requiring you to set up and maintain infrastructure for hosting them. You don't need **Flask** or **FastAPI** with API Gateway because it handles all the API-related tasks (like routing, security, and scaling) that those frameworks typically manage in a self-hosted environment.

Example with Flask:

```python
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Run inference
    prediction = model.predict(data['input'])
    return jsonify({'prediction': prediction.tolist()})
    
app.run(port=5000)
```

Example with API Gateway and Lambda:
```python
import json
def lambda_handler(event, context):
    data = json.loads(event['body'])
    prediction = model.predict(data['input'])
    return {
        'statusCode': 200,
        'body': json.dumps({'prediction': prediction.tolist()})
    }
```
* the `lambda_handler` function doesn’t directly specify HTTP methods (like `POST`, `GET`) or endpoint paths. Instead, this is configured **outside the Lambda function** in **API Gateway**.
* The HTTP method is specified in the Gateway console:
	* Create an API `MyInferenceAPI` and define a resource path (e.g., `/predict`).
	- Attach an HTTP method (e.g., `POST`) to this resource.
	- Link the HTTP method to your Lambda function.
- The client (e.g., Postman, Python script, or frontend) interacts with the endpoint generated by API Gateway:
	- **URL**: `https://<api-id>.execute-api.<region>.amazonaws.com/<stage>/predict`
		- `<api-id>`: Unique identifier for your API.
        - `<region>`: AWS region (e.g., `us-east-1`).
        - `<stage>`: Deployment stage (e.g., `dev`, `prod`).

A/B Testing with AWS Lambda. Two options:
- Inside the lambda function we randomly assign one of two models
- At the Gateway API level we split the traffic.
# Protobuf

**Protocol Buffers (protobuf)** can be used to define the structure of the **invocation** and **return payloads** for an API or service. This is common when you want to ensure that all communication between systems (such as microservices or between a client and a server) follows a strict, well-defined format for the data being exchanged. 

Protocol Buffers (protobuf) is a language-agnostic data serialization format developed by Google. It is used for efficiently serializing structured data, which is often used in communication protocols, data storage, or APIs.
-  **Serialization**: The process of converting data into a format (like a byte stream) that can be transmitted over a network or stored.
-  **Schema**: Protobuf uses a schema, defined in `.proto` files, to describe the structure of the data, including the types of fields, field numbers, and how they should be serialized.

Example `.proto` file:

```protobuf
syntax = "proto3";

package messages;

// Define a request message
message Request {
    string user_id = 1;
    string query = 2;
}

// Define a response message
message Response {
    int32 status_code = 1;
    string message = 2;
    string data = 3;
}
```

- **Invocation Payload**: The data sent when calling the service (e.g., making a request). This is usually the input data to the API or function you are calling.
- **Return Payload**: The data returned after the service completes processing the request. This is typically the response sent back to the caller.

Once you have your protobuf file (e.g., `messages_common.proto`) defining the invocation and return payloads:
1. **Generate Code from the Protobuf**: Protobuf can generate code for multiple languages (e.g., Python, Java, Go, etc.) from the `.proto` files.
2. **Integrate Into Your Service**:
	1. In the **request handler** (e.g., Lambda, API Gateway, microservice), you would deserialize the **incoming request** payload using the generated code,
	2. In the **response handler**, you would serialize the return data into the `ReturnResponse` format, which will be sent back to the client.

```python
import json
import messages_common_pb2  # Import the generated protobuf file

def lambda_handler(event, context):
    # 1. Deserialize the incoming request body using protobuf
    invocation_request = messages_common_pb2.InvocationRequest()
    invocation_request.ParseFromString(event['body'])

    # 2. Process the request (e.g., ML model inference or other processing)
    result_data = f"Result for query: {invocation_request.query}"

    # 3. Create a response (protobuf-encoded)
    return_response = messages_common_pb2.ReturnResponse(
        status_code=200,
        message="Success",
        result_data=result_data
    )

    # 4. Serialize the response to send back to the client
    response_body = return_response.SerializeToString()

    # 5. Return the response to API Gateway (must be a JSON object with 'body')
    return {
        'statusCode': 200,
        'body': response_body,
        'isBase64Encoded': True  # Indicate that the body is binary (protobuf)
    }
```

Why use protobuf?
- is a **standardized format** that is language-agnostic and platform-independent. By using protobuf, you're following a widely adopted industry standard for defining structured data. This allows systems written in different languages (like Python, Java, Go, C++, etc.) to communicate easily using the same well-defined format.
- **Protobuf** serialization is designed to be **highly efficient** in terms of both **space** and **speed**.
- built-in **validation** for types and required fields.
- built in conversion to any language

Takeaway: It's a way to **serialize** and **validate** the input and output data in APIs (including REST APIs like Flask, FastAPI, or AWS Lambda).

Efficiency note: warm starts mean that potentially some parts of the code can be shared between several calls e.g. model initialization which can take time

```python
import json
model = load_model_once()  # Load model once at cold start and reuse for subsequent invocations
cache = {}  # Dictionary to cache results

def lambda_handler(event, context):
    data = json.loads(event['body'])
    input_data = data['input']

    # Check if prediction for the input is cached
    if input_data in cache:
        prediction = cache[input_data]
    else:
        prediction = model.predict(input_data)
        cache[input_data] = prediction  # Cache the prediction

    return {
        'statusCode': 200,
        'body': json.dumps({'prediction': prediction.tolist()})
    }

```
# Caching
The main goal of caching is to avoid recomputing the results for the same inputs, which can save time, resources, and reduce costs.

## AWS Lambda

AWS Lambda functions are stateless, meaning they do not retain any memory of previous invocations once the function execution is complete. However, caching can still be implemented in Lambda using several strategies:
* **Warm Starts**: Lambda functions that are invoked repeatedly within a short period may be reused (i.e., the environment doesn't need to be reinitialized). The model can also be just initialized once.
* **Use external cache Amazon ElastiCache (redis/memcached)**

```python
import json
import redis

r = redis.StrictRedis(host='your-cache-cluster-url', port=6379, db=0)

def lambda_handler(event, context):
    data = json.loads(event['body'])
    input_data = data['input']

    # Check if prediction for the input is cached in Redis
    cached_prediction = r.get(input_data)
    if cached_prediction:
        prediction = json.loads(cached_prediction)
    else:
        prediction = model.predict(input_data)
        r.set(input_data, json.dumps(prediction))  # Cache the result in Redis

    return {
        'statusCode': 200,
        'body': json.dumps({'prediction': prediction.tolist()})
    }

```

## AWS EC2
When deploying a model on **AWS EC2**, you have more control over the environment since EC2 instances are not as stateless as Lambda. You can implement caching in a few different ways:
- You can use **in-memory caches** like Python dictionaries or **Redis/Memcached** for faster retrieval of previously computed predictions.
- Since EC2 instances persist over time, you can cache the results in the memory of the instance, so that you don't need to recompute predictions for the same input during the lifespan of the instance.

## AWS EKS

With **AWS EKS**, you deploy machine learning models as **containers** inside Kubernetes clusters. Caching in EKS can be implemented in several ways, depending on the architecture:
- Use **ElastiCache (Redis/Memcached)** for distributed caching that can be accessed by multiple pods (containers) running in the EKS cluster.
- in-memory caches like a python dictionary.

## Redis and Memcached

![Pasted image 20241220134109](/assets/Pasted%20image%2020241220134109.png)
**Redis** and **Memcached** are **not exclusive to AWS**. They can be run on any infrastructure, whether on-premises, on virtual machines, or in the cloud. However, AWS provides managed services for both of them, making it easier to set them up and scale in a cloud environment

**AWS ElastiCache**: This is a fully managed service for both Redis and Memcached provided by AWS. You can use it to set up, operate, and scale these data stores without worrying about managing servers and other infrastructure details.

### Practicalities
**Set up ElastiCache Cluster:**
- **Navigate to AWS ElastiCache**: Go to the AWS Management Console and search for **ElastiCache**.
- **Choose Redis or Memcached**: You can choose either Redis or Memcached depending on your use case.
- **Create a Cluster**: Follow the wizard to configure your cluster (choose the type, number of nodes, security group, etc.).
- **Security**: Ensure that you configure **VPC** and **security groups** properly so that your application can access the cache. ElastiCache will give you an **endpoint** for connecting to the cache.
**Connecting to ElastiCache (Redis or Memcached) from Your Application**:
- **Redis**: Use the **`redis-py`** library to connect to Redis.
```python
import redis
r = redis.StrictRedis(host='your-elasticache-endpoint', port=6379, db=0)
r.set('key', 'value')
print(r.get('key'))
```
- **Memcached**: Use the **`pymemcache`** library for Memcached.
```python
from pymemcache.client import base
client = base.Client(('your-elasticache-endpoint', 11211))
client.set('key', 'value')
print(client.get('key'))
```

the actual query to the cache and the update of it can happen inside the API when handling a machine learning (ML) model prediction request.

```python
import redis
import json
from flask import Flask, request

# Setup Flask app and Redis connection
app = Flask(__name__)
r = redis.StrictRedis(host='localhost', port=6379, db=0)

# Example ML model (replace with your actual model)
def predict(input_data):
    return {"prediction": "some result based on input"}

@app.route('/predict', methods=['POST'])
def predict_api():
    # Get input data from request
    data = json.loads(request.data)
    input_data = data.get('input')

    # Check if the result is cached
    cached_result = r.get(input_data)
    if cached_result:
        # Return cached result if it exists
        return cached_result

    # If not cached, run the model and cache the result
    prediction = predict(input_data)

    # Store the result in the cache (set a timeout of 1 hour for example)
    r.setex(input_data, 3600, json.dumps(prediction))

    # Return the prediction
    return json.dumps(prediction)

if __name__ == '__main__':
    app.run(debug=True)
```

Note: **Redis** is often used as a caching mechanism, but it's actually a **data store** that can be used in several different ways, not just for caching. Redis is an **in-memory database**, meaning it stores data in memory (RAM) rather than on disk like traditional databases. This makes it extremely fast for operations, which is why it is commonly used for caching, session management, real-time data processing, and pub/sub messaging.
# AWS deployment choices

| **Deployment Option**   | **Best For** | **Key Features** | **Pros** | **Cons** |
|-------------------------|--------------|------------------|----------|----------|
| **AWS Lambda** | Small-scale, event-driven ML inference | Serverless, event-based, auto-scaling | No server management, cost-efficient, scales automatically | Cold starts, execution time limits (15 mins), limited resources (memory/storage) |
| **AWS EC2** | More complex or larger-scale applications | Full control over infrastructure, flexible | Custom environment, persistent resources, more flexibility | Management overhead, cost (pay for uptime), need to scale manually |
| **AWS Kubernetes (EKS)** | Large-scale containerized applications | Orchestrates containerized applications, auto-scaling | Flexibility, scaling, and orchestration of containers | Complex setup, infrastructure management, cost of running clusters |
| **AWS SageMaker** | End-to-end ML lifecycle management | Fully managed service for training, tuning, and deployment | End-to-end ML solution, auto-scaling, built-in monitoring and logging | Can be more expensive, may have more overhead for small projects |
| **Google Cloud AI / Vertex AI** | End-to-end ML lifecycle management (non-AWS) | Similar to SageMaker, fully managed ML lifecycle | End-to-end solution, easy deployment | Not AWS-specific, different platform and pricing |
| **Azure ML** | End-to-end ML lifecycle management (non-AWS) | Similar to SageMaker, fully managed ML lifecycle | End-to-end solution, easy deployment | Not AWS-specific, different platform and pricing |

## Fully managed services

Companies may choose **fully managed services** like **SageMaker** or **Vertex AI** over **EKS** or **EC2** when they want to reduce operational complexity, leverage automation and advanced ML features, and focus on their core ML tasks without worrying about managing the underlying infrastructure. **EKS** and **EC2** are more suitable for teams that need more control over their infrastructure or have very specific deployment requirements.
## Vertex AI
**Vertex AI** is Google Cloud's unified artificial intelligence (AI) and machine learning (ML) platform designed to simplify the process of building, deploying, and managing machine learning models. It offers a comprehensive suite of tools and services to handle all aspects of the ML lifecycle, from data preparation and model training to deployment and monitoring.
**Vertex AI** is Google Cloud's equivalent to **AWS SageMaker**, offering similar functionalities like model training, deployment, and monitoring
# Specification and documentation of APIs
**OpenAPI**, also known as **Swagger** (the name of the specification before it became OpenAPI), is a specification for describing and documenting RESTful APIs. It's a standardized way to describe the structure, inputs, outputs, and operations of an API in a machine-readable format (usually YAML or JSON). It provides a clear way for developers and systems to interact with APIs by defining things like endpoints, parameters, request/response formats, authentication, and more.

# Authentication

**Authentication** ensures that only authorized parties (users, systems, or services) can interact with your ML model. In AWS, authentication is typically managed using **AWS Identity and Access Management (IAM)**, **API keys**, or **AWS Signature Version 4 (SigV4)**, depending on the service being used.
## Aws signature version - SigV4
- **SigV4** is an authentication mechanism used to sign API requests made to AWS services. When a client (like a robot or application) sends an HTTP request to an ML service (e.g., **Amazon SageMaker** or **AWS Lambda** for inference), the request is signed using AWS credentials (e.g., `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`).
- This signing process includes creating a cryptographic signature based on the request, time, and credentials, which is then included in the request header. The AWS service receiving the request uses this signature to verify that the request has not been tampered with and that it comes from a legitimate, authenticated source.

*e.g. in a **REST API** using **SigV4** authentication, **each individual request** must be signed with valid credentials, and the **authentication process is typically handled at the API level**. The authentication process ensures that every incoming request is properly validated before it can reach the actual API implementation.

- **Authentication** happens in the **API layer**, often using **AWS API Gateway** or custom server logic to validate the request’s signature before reaching the core business logic.
- **Client signs** the HTTP request using **SigV4** (AWS credentials + hashing).
- - **Server** (API Gateway or your service) **validates** the signature by recreating the canonical request and checking the signature against the stored secret key.

# Data pipelines

 A **data pipeline** refers to a series of processes or steps that automate the collection, transformation, and feeding of data into ML models for training or inference. A **data pipeline** is designed to manage and streamline the flow of data, ensuring that it is properly pre-processed, cleaned, transformed, and formatted before being input into a machine learning algorithm.

Key components:
- **Data Collection**: Gathering data from various sources, such as databases, APIs, or external sources (e.g., auction data, car images).
- **Data Cleaning & Preprocessing**: Handling missing values, encoding categorical variables, normalizing numeric features, and any other necessary transformations.
- **Feature Engineering**: Creating new features from raw data, which can include aggregation, feature selection, or time-based transformations.
- **Model Training**: Feeding the cleaned and preprocessed data into an ML algorithm for training.
- **Model Evaluation**: Evaluating the trained model’s performance (e.g., through validation sets, cross-validation).
- **Model Deployment (In Production)**: The trained model is deployed and used to make predictions on new, incoming data (e.g., predicting car prices or recommending vehicles).
- **Monitoring & Retraining**: Continuously monitoring model performance and retraining the model as new data comes in.
## Why are data pipelines important?
- **Automation**: They help automate repetitive tasks like data cleaning, feature engineering, and model training.
- **Scalability**: ML models often require large volumes of data. Pipelines allow processing and feeding this data efficiently.
- **Consistency**: A pipeline ensures that the same transformations are applied to both the training and real-time (inference) data, ensuring consistency in model performance.
For example, in a **car auction platform** where models are predicting car prices, a pipeline might collect auction data, transform it into features (such as the car's age, mileage, brand, etc.), feed the data to a pricing model, and provide real-time predictions during auctions.

## Tools
### Apache AIRFLOW

**Apache Airflow** is an open-source platform used to **programmatically author**, **schedule**, and **monitor workflows**. Workflows are represented as Directed Acyclic Graphs (DAGs), which define the tasks and their dependencies. Airflow can be used to automate and manage workflows like data extraction, transformation, loading (ETL), or even machine learning model training and deployment.

In the context of **machine learning**, Airflow can be used for:

- Automating the end-to-end machine learning pipeline.
- Moving and transforming data between different services (e.g., from a database to a model training environment).
- Scheduling tasks (e.g., retraining models periodically).
- Managing complex dependencies between tasks (e.g., data preprocessing before model training).

Imagine you're working for a car auction website, and you want to automate a workflow that:

1. Fetches auction data daily.
2. Preprocesses the data.
3. Retrains a machine learning model.
4. Deploys the model.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime, timedelta

# Define the Python functions for each task
def fetch_data():
    print("Fetching auction data...")

def preprocess_data():
    print("Preprocessing auction data...")

def train_model():
    print("Training machine learning model...")

def deploy_model():
    print("Deploying model to production...")

# Define the default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 12, 20),
    'catchup': False  # Don't backfill missing runs
}

# Define the DAG
dag = DAG(
    'ml_model_training',
    default_args=default_args,
    description='A simple ML pipeline to fetch, preprocess, train, and deploy a model',
    schedule_interval=timedelta(days=1),  # Run daily
)

# Define the tasks and their dependencies
start = DummyOperator(
    task_id='start',
    dag=dag,
)

fetch_task = PythonOperator(
    task_id='fetch_data',
    python_callable=fetch_data,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

deploy_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag,
)

# Task dependencies: Execute in this order
start >> fetch_task >> preprocess_task >> train_task >> deploy_task

```
Once this code is saved in an Airflow DAG folder, you can:
- Start the Airflow web UI (typically at `http://localhost:8080`).
- Trigger the DAG manually or let it run based on the defined schedule.
### When is this necessary?

In many cases, especially when working with smaller datasets or with less frequent training needs, a manual approach (such as SSH-ing into a machine and running scripts) is sufficient. However, as companies scale and the complexity of data and workflows increases, automated and orchestrated pipelines like Airflow can bring significant benefits.

For larger companies or systems where training or data processing needs to happen **on a schedule** (daily, weekly, etc.), automating these tasks using orchestration platforms ensures that everything runs without human intervention. For example, fetching data from a remote server or cloud storage, preprocessing, and retraining a model daily can be fully automated, freeing up resources for more important tasks.
## Scikit-Learn pipelines
In **Scikit-Learn**, the `Pipeline` is a class that helps encapsulate the process of transforming data and applying machine learning algorithms in a linear sequence.

Chain together different preprocessing steps (e.g., scaling, encoding) with model training steps (e.g., fitting a regression model).

Example where we predict **house prices** based on features such as **size**, **number of rooms**, and **location**.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Define the preprocessing steps
# - For numeric columns: Impute missing values and scale the data
# - For categorical columns: Impute missing values and apply one-hot encoding

numeric_features = ['size', 'num_rooms', 'age']
categorical_features = ['location']

# Preprocessing for numeric data: Impute missing values, then scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Replace missing values with the mean
    ('scaler', StandardScaler())  # Standardize the data
])

# Preprocessing for categorical data: Impute missing values, then one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Replace missing values with the most frequent
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
])

# Combine both preprocessing steps into a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Step 4: Define the pipeline that includes the preprocessor and the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))  # Use RandomForest for regression
])

# Step 5: Train the model
pipeline.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = pipeline.predict(X_test)
```
# Glossary

**SaaS (Software as a Service) environment in machine learning** refers to cloud-based platforms or services that provide tools, infrastructure, and frameworks to develop, deploy, and manage machine learning (ML) models. These platforms eliminate the need for users to set up and maintain their own hardware or software infrastructure, enabling easier access and faster adoption of ML technologies.

**A/B** testing is used to evaluate and compare the performance of different models:
- Compare a new model (B) against the current production model (A) to determine if the new model offers better performance on key metrics.
- It may involve
	- Run both models simultaneously in production, serving different versions to different user groups. Perhaps a small % to B.
	- Assign users randomly to either Model A (control) or Model B (variant). This ensures that external factors are evenly distributed between groups.
	- **Define Metrics**: such as
	    - **Accuracy**: Percentage of correct predictions.
	    - **Conversion Rate**: Users taking a desired action (e.g., purchases).
	    - **Engagement**: Time spent on a platform.
	    - **Revenue**: Total income generated.
	- Log these metrics both for A and B
	- Analyse the results, perhaps using hypothesis testing or just comparing stats.
	- Decide whether to keep A or B.


**Amazon CloudWatch Evidently** is a feature of Amazon Web Services (AWS) that enables **feature flagging** and **experimentation** for applications. It helps developers and teams roll out new features in a controlled manner, test different variations of a feature, and monitor how these features perform in real-time. Essentially, it is a tool for **feature management**, **A/B testing**, and **experimenting with features** within applications.

API throttling is the process of limiting the number of API request a user can make in a certain period.