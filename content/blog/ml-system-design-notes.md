---
has_been_reviewed: false
tag: Machine Learning Engineering
tags:
  - Computer
  - Vision
aliases: 
publish: false
slug: ml-system-design-notes
title: Notes on ML system design
description: Assorted notes and takeaways from discussions
date: 2024-11-26
image: /thumbnails/backbones.png
---
# What happens after you train a ML model

Let's walk through a generic scenario. You just trained your Pytorch model, and you want to deploy it to production.

## 1- Prepare the model for inference

Load the checkpoint file and model definition ; switch the model to evaluation mode (disables dropout and batchnorm updates)

```python 
import torch from my_model import MyModel  Load the model model = MyModel() checkpoint = torch.load("path/to/your/checkpoint.pth", map_location=torch.device('cpu')) model.load_state_dict(checkpoint) model.eval() 
```

## 2 - (Optional)- Quantization

Quantization reduces the precision of a model's parameters (e.g., from 32-bit floating-point numbers to 8-bit integers). The goal is to make the model smaller, faster, and more efficient, especially on hardware like CPUs, GPUs, or edge devices.

### When does it happen?
1. **Post-Training Quantization** (Most Common): - This happens after training is complete, during the model deployment phase. - You take the trained weights (e.g., in FP32) and convert them to a lower precision (e.g., INT8 or FP16). - This can be done **before or after exporting to ONNX**, depending on the deployment pipeline. 
2. **Quantization-Aware Training** (More Advanced): - In this approach, quantization is simulated during training. The model learns to adapt to the lower precision, resulting in better accuracy after quantization. - This is suitable for applications where accuracy loss from standard post-training quantization is unacceptable. - Happens during the training phase but requires additional setup.

### Should you quantize?
Quantization is **not always necessary**. It depends on the use case and constraints: - 
**When to Use Quantization**: - Deploying on hardware with limited resources (e.g., edge devices, mobile phones). - Optimizing for low latency or reduced power consumption. - When slight reductions in model accuracy are acceptable. - 

**When Not to Use Quantization**: - If your application runs in environments with sufficient computational power and memory (e.g., cloud GPUs). - If even minor accuracy losses are critical (e.g., in medical or safety-critical systems). - If your model relies heavily on operations that aren't well-supported by quantization (e.g., certain custom layers)

## 3- (Optional) Optimize the model exporting to TorchScript or ONNX

Exporting your model to formats like ONNX (Open Neural Network Exchange) or TorchScript can make it easier to deploy and run your model efficiently across different platforms. Let’s dive into the details.

### What is ONNX? 
ONNX is a **standard format** for machine learning models, designed to allow models trained in one framework (e.g., PyTorch, TensorFlow) to be run in another framework or specialized runtime environments (like ONNX Runtime). It's widely supported by various platforms and tools.

### Why Export to ONNX or TorchScript? 
**ONNX**: Good for cross-platform compatibility. You can train a model in PyTorch and deploy it using ONNX Runtime, TensorRT, or even on edge devices like mobile phones. 
**TorchScript**: Optimized specifically for PyTorch. It's portable and can be run without needing the Python interpreter (e.g., in C++ apps, mobile).

Both formats aim to make deployment easier, with ONNX focusing on broad compatibility and TorchScript prioritizing PyTorch-centric workflows. Which one you choose depends on your deployment needs. 
### How to Export to ONNX? 
PyTorch makes it relatively straightforward to export a model to ONNX. Here’s how: 

1. **Prepare Your Model for Inference**: Same as before, load your checkpoint and switch to evaluation mode. 
2. **Define a Dummy Input**: ONNX requires a dummy input to trace the model’s computation graph. 
3. **Export**: Use `torch.onnx.export` to save the model in ONNX format.

```python 
import torch from my_model import MyModel #

# Load and prepare the model 
model = MyModel() checkpoint = torch.load("path/to/your/checkpoint.pth", map_location=torch.device('cpu'))

model.load_state_dict(checkpoint) 
model.eval() 

# Define a dummy input (adjust dimensions as needed) 
dummy_input = torch.randn(1, 3, 152, 152) 
# Export to ONNX 
torch.onnx.export( model, # The model dummy_input, 
				   dummy_input, # Dummy input
				   "model.onnx", # Output file name
					export_params=True, # Save weights inside the model file
					opset_version=11, # ONNX version to target
					...)
```

### Convert to ONNX/Torchscript or keep the original model
In production, whether or not you convert your model to ONNX, TorchScript, or another format depends on **how and where the model will be deployed**. 

Conversion is common in production **when performance, portability, or platform compatibility matter**. But if the deployment stack fully supports the native framework and performance is acceptable, you can skip it. It’s more of a "best practice" than a strict requirement.

### Optimize
- Use an inference runtime like **ONNX Runtime**, TensorRT (for NVIDIA GPUs), or OpenVINO (for Intel hardware). -
- These tools optimize the ONNX model by fusing operations, removing redundancies, or tailoring it to specific hardware. -
- This step ensures that the model runs as efficiently as possible on the target infrastructure.

## 4- Deployment

In order to make the model available to potentially millions of users, we need to choose the deployment infrastructure. 

If the model is going to run on e.g. each edge device, deploy the quantized model to edge devices using ONNX Runtime Mobile or TensorFlow Lite.

If the model is going to run on a server, we need to set up a prediction server.

### 4.1 - API and containerization

- Write a FastAPI application exposing `/predict` endpoints. 

Alternatives: Flask, FastAPI, gRPC, REST.  **TorchServe**, **TFServing**
Optimize API endpoints to minimize latency (e.g., batching requests when possible, caching results). Expose the model’s predictions via an API

- Package the app and ONNX model in a Docker image.

Docker is a tool that lets you package an application (like your model) and everything it needs (code, libraries, dependencies, configurations) into a **container**. A container is a lightweight, portable, and self-sufficient unit that can run consistently across different environments—on your laptop, in the cloud, or on a production server. Think of it as a **virtual machine**, but much faster and more efficient because it shares the host operating system.

Why use docker?
- **Consistency**: Your model and app will work the same way everywhere, regardless of the host machine. - 
- **Portability**: You can move your container across machines or cloud providers. - 
- **Simplicity**: All dependencies are bundled together, avoiding the classic “it works on my machine” problem. - 
- **Scalability**: Containers are lightweight, so you can spin up multiple instances to serve millions of users.

How Does Docker Work? 
1. **Dockerfile**: This is a text file with instructions to build your container (like a recipe). 
2. **Docker Image**: When you run the Dockerfile, it creates an image—a snapshot of your app, libraries, and environment. 
3. **Docker Container**: A container is a running instance of the image.

#### Steps to Put Your Model into a Docker Container 
1. **Prepare Your Application**: - You’ll need your model (e.g., ONNX file), your prediction code (Python script or FastAPI/Flask app), and a `requirements.txt` file listing the Python libraries. 
2. **Write a Dockerfile**: - A Dockerfile tells Docker how to build the container. Here’s an example: 

```dockerfile 
# Use an official Python image as the base FROM 
python:3.9-slim 

# Set the working directory in the container 
WORKDIR /app 

# Copy the application code and model files 
COPY . /app 

# Install required Python libraries 
RUN pip install --no-cache-dir -r requirements.txt 

# Expose the port your API will run on (optional, helps with networking) 
EXPOSE 8000 

# Command to run your app (e.g., a FastAPI server) 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] ``` 

Key lines: 
- `FROM python:3.9-slim`: Starts with a lightweight Python image. 
- `COPY . /app`: Copies your app’s files (model, code, etc.) into the container. 
- `RUN pip install`: Installs dependencies inside the container. - `CMD`: Runs your app when the container starts. 

3. **Build the Docker Image**: - Open a terminal in the folder with the Dockerfile and run: ```bash docker build -t my-model-api . ``` - This creates an image named `my-model-api`. 
4. **Run the Docker Container**: - Start the container with: ```bash docker run -d -p 8000:8000 my-model-api ``` - `-d`: Runs the container in the background. - `-p 8000:8000`: Maps port `8000` on your machine to port `8000` in the container. Your API should now be accessible at `http://localhost:8000`. 
5. **Test Your API**: - Use `curl`, Postman, or your browser to make requests to the API and see if it works. 

### 4.2 Deploy

The next step is to deploy it, especially when you need to serve **thousands or millions of clients**.

### 4.2.0 - Push to a model registry

A *registry* is essentially a database or storage system that keeps track of models, their versions, metadata (like hyperparameters and metrics), and their status (e.g., "staging," "production")

### 4.2.1 - Container Orchestration
For large-scale deployment, **container orchestration tools** (aka container orchestration platform) help you manage your application efficiently. Popular tools are **Kubernetes** and **Docker Swarm**

#### Kubernetes
- Kubernetes helps **automate** the deployment, scaling, and management of containerized applications. - It manages clusters of containers across multiple machines. 
- **Pods** in Kubernetes are the basic unit of deployment. A pod can contain one or more containers (e.g., your FastAPI app in a Docker container). 
- Kubernetes can automatically **scale** your containers up or down depending on traffic (i.e., adding more containers as demand grows). 
- **Services** in Kubernetes expose your app to the outside world, handle load balancing, and route traffic to the right containers. 
Example flow: 
1. Your application is wrapped in a Docker container (ONNX model + API service). 
2. You push the Docker image to a container registry like **Docker Hub**, **Google Container Registry (GCR)**, or **AWS Elastic Container Registry (ECR)**. 
3. You define a **Kubernetes Deployment** that tells Kubernetes how to deploy your Docker image, and a **Kubernetes Service** that exposes your application to external traffic
4. Kubernetes handles scaling, load balancing, and routing based on the traffic load.

Takeaway: Container Orchestration (e.g. Kubernetes) is  layer of the infrastructure that helps manage, scale and deploy containerized applications, automating tasks such as **scaling**, **load balancing**, **self-healing** (restart failed nodes). This layer takes care of the **operational side** of **running containers at scale**.

### 4.2.2 - Cloud providers

Once you have Kubernetes (or Docker Swarm), you need a platform to run it on. These are some of the common environments for running containerized applications: 
*  **Amazon Web Services (AWS)**: 
	* AWS offers managed Kubernetes through **Amazon EKS** (Elastic Kubernetes Service) and Docker containers with **Amazon ECS** (Elastic Container Service).  
	* Amazon also offers EC2 **Elastice Compute Cloud**, virtual machines where you can run applications including Kubernetes clusters.
* **Google Cloud Platform (GCP)**: GCP has **Google Kubernetes Engine (GKE)** for managing Kubernetes clusters and **Google Cloud Run** for deploying containerized applications without needing to manage the infrastructure.
*  **Microsoft Azure**: Azure provides **Azure Kubernetes Service (AKS)** to manage your Kubernetes clusters and **Azure Container Instances** for simpler container deployments. 
These platforms make it easy to **deploy**, **manage**, and **scale** your application across multiple machines or across different data cente

Takeaway: Cloud providers like AWS, Google Cloud or Azure offer **infrastructure**, they provide the **underlying servers** (virtual machines) where Kubernetes run.  So, **Kubernetes** helps you manage your containers and services, while the **cloud provider** gives you the machines (compute power) to run them.

In summary: **Kubernetes** handles how to run and manage your containers, while **the cloud** ensures you have the infrastructure (compute, networking, storage) to run them at scale. They complement each other.
### Concrete example
(a) Choose a Kubernetes Service
Kubernets is open-source. You typically use a **managed Kubernetes service** from a cloud provider. For example, **AWS Elastic Kubernetes Service (EKS)** simplifies managing Kubernetes clusters on AWS.
- **AWS EKS**: You would create a Kubernetes cluster in EKS. This is where you’re setting up your “container orchestration.” - 
- **Cost**: For EKS, AWS charges based on the number of Kubernetes clusters you have, and for each worker node in the cluster (these are the virtual machines or EC2 instances that will run your containers).
(b) Configure Kubernetes
Once EKS is set up, you: 
- Configure your Kubernetes cluster to pull the Docker image from the registry (e.g., from AWS ECR or Docker Hub). 
- You define a Kubernetes **deployment** (this is essentially the setup that tells Kubernetes how many containers of your model to run, their configuration, resource allocation, etc.). 
- You also configure **services** (to expose the model API to the outside world), **ingress** (for more advanced traffic routing), and other Kubernetes resources to manage scaling, load balancing, etc. 

**Tools for configuration**: Most of this is done by writing **YAML files** that define the Kubernetes resources (like deployments and services). You’d then use the `kubectl` command to apply these configurations to the cluster.
(3) Deploying to AWS Elastic Compute Infrastructure

Behind the scenes, Kubernetes requires **compute resources** (virtual machines) to run the containers. This is where **AWS EC2** comes into play. These virtual machines are where your containers will be running. With managed Kubernetes like **EKS**, AWS handles provisioning EC2 instances for you, but you still need to configure the types and sizes of these instances. You don’t really have to manually create EC2 instances — **EKS automatically does this for you** based on the configurations you set for your Kubernetes cluster (e.g., instance types, number of instances). 

Autoscaling** You can configure **auto-scaling** on your EC2 instances. **AWS Elastic Load Balancer (ELB)** is often used here to distribute incoming traffic to your EC2 instances. 

Your bill at the end of the month would be based on: 
- How many EC2 instances were running and for how long. 
- How much traffic went through your load balancer. 
- How much storage you used for your container images (e.g., ECR). 

Takeaway:
- You upload your Docker image to a container registry (e.g., AWS Elastic Container Registry or Docker Hub). 
- You configure Kubernetes with a YAML file that defines how many replicas of your service to run, resource limits (CPU, memory), and other settings. 
- Kubernetes pulls the Docker image from the registry and deploys it across the cloud provider’s servers.

### Considerations:

To handle large volumes of users, you need to automatically scale your application and manage the incoming traffic. 

- **Load Balancer**: When you expose your service (e.g., FastAPI or Flask) to external traffic, you typically place a **load balancer** in front of it. This can distribute requests across multiple instances of your containers to ensure **even distribution of traffic** and prevent any one container from being overwhelmed. - Cloud services like AWS, GCP, and Azure provide **managed load balancers** that you can configure to work with Kubernetes or Docker Swarm. 
- **Auto-Scaling**: Both Kubernetes and cloud providers support **auto-scaling**, where the system automatically increases the number of running containers when demand spikes and reduces it when traffic decreases. - Kubernetes, for instance, can scale the number of pods based on CPU and memory usage or based on custom metrics (like request count per minute).
- A **URL or endpoint** (e.g., `https://roomba-enhance-map.com`) is assigned to your service, making it accessible. roombas will send their maps to be precesed by a ML algorithm there.

Technologies:
 Kubernetes, AWS SageMaker, Google AI Platform.
**scalable backend** (e.g. AWS EC2, GCP Compute Engine or Kubernetes CLUSTERS)

More:
* Deploy a **CDN** if the model needs to serve geographically distributed users. If you're serving large media files or static content, a **CDN** (like **Cloudflare**, **AWS CloudFront**, or **Google Cloud CDN**) can distribute the content across geographically distributed servers, reducing latency.
* Implement **caching** layers (e.g., Redis, Memcached) for repeated queries or frequently used predictions to reduce load on the model.
* **Security** is crucial when serving millions of users:
	* API Gateway: To impose a limit to the rate of queries of a user, authentiction, etc.


### Summary:

Once you have the Docker image: 1
1. **Push it to a container registry** (e.g., Docker Hub, AWS ECR). 
2. **Deploy it using Kubernetes or Docker Swarm** in a cloud or on-premise infrastructure.  E.g. Kubernetes + Amazon EC2
3. **Use load balancers** to distribute traffic and **auto-scaling** to handle variable loads. 
4. Set up **CI/CD pipelines** for automatic updates and deployments. 
5. **Monitor and log** the system for potential issues. 
6. Use **caching** and **CDNs** to improve performance at scale. 
7. **Secure** your API and data with proper **authentication**, **rate limiting**, and **encryption**.

## Monitoring
 - Track key metrics like API latency, throughput, error rates, and model inference times.  Use tools like **Prometheus** and **Grafana** for monitoring, or cloud-native monitoring solutions (e.g., AWS CloudWatch).
 * Use tools like **Prometheus** and **Grafana** for monitoring, or cloud-native monitoring solutions (e.g., **AWS CloudWatch**).
 * Implement **logging** to track predictions, errors, and unusual behavior.
 * track shifts in data distribution (input or predictions)

## CI/CD ; Data pipeline for retraining

In a real production environment, you'll likely want to update your models and API without downtime. This is where a **Continuous Integration/Continuous Deployment (CI/CD)** pipeline comes in. 

Continuous Integration (CI) and Continuous Deployment (or Delivery) (CD) are practices that ensure your entire pipeline—training, testing, deploying, and maintaining machine learning models—remains smooth, automated, and efficient.

A CI/CD pipeline automates the process of: 
- **Building** the Docker image when changes are made (e.g., updating the model, changing the code, etc.). 
- **Testing** the image to ensure everything works. 
- **Pushing** the updated image to the registry. 
- **Deploying** the new image to your Kubernetes or Docker Swarm cluster. 

Popular CI/CD tools: 
* Version control systems: GitHub / GitLab / Bitbucket
	* **Code**: Store your Python scripts, model definitions, and API code in repositories.
	*  **Data and Artifacts**: Use tools like **DVC (Data Version Control)** or store raw data in an object store (e.g., AWS S3) while tracking it via Git.
* **Github Actions**,  **GitLab CI/CD**, or **Jenkins** can be used to define workflows, such as "When a team member commits new preprocessing code, GitHub Actions runs:  Unit tests + Integration tests+ Model validation tests"
* **Model Artifacts (Storage and versioning) tools**:
	* MLFlow: Tracks and stores model versions, models themselves, metadata, artifacts. 
	* W&B: Tracks experiments and training runs.
	* Object Storage: AWS S3 / Google Cloud Storage / Azure Blob Storage: Store large models

Takeaway: You can use **CI/CD pipelines** to automatically update your Docker image, push it to the container registry, and update your Kubernetes deployment.
# Prominent tools and technologies

## MLFlow
About MLFlow and what it can do, which comprises several of the stages described above:
* Store details during training runs, and allow comparison of them (similar to W&B)
* Model registry: Stores models and their metadata
* Package of models into standard format e.g. ONNX
* Provides built-in tools to serve the model as a REST API
* W&B is fantastic for the *research phase* of machine learning, while MLflow excels when moving models from research to production.

# Amazon SageMaker
If you're using SageMaker, it can handle many of the steps we’ve been discussing. For example: 
- You might prepare and preprocess your data in SageMaker.
- You might train and evaluate a model in SageMaker. 
- Register the model in SageMaker's **Model Registry**. 
- Deploy the model using SageMaker-hosted endpoints. 
- Monitor the performance, retrain, and redeploy—all in SageMaker. 
It can integrate with Docker and Kubernetes if you're using custom workflows, but it's designed to minimize the need for those extra tools by managing everything for you.


# 5- Toy example, local server, if you don't care about scalability and it's not a product

"Put a model behind an API": Wrapping the model in a service that takes input (like an image or data) and returns predictions (like a label or value). This makes it accessible to others.

* **API**: A doorway others can use to send requests (data) and get responses (predictions).
* **Prediction Service**: A program running on a server that loads the trained model and waits for incoming requests via the API. It processes the input, runs it through the model, and sends back the result.



You’ll need to set up an **endpoint** to receive input (like images), process them, and return predictions.

```python 
from fastapi import FastAPI, File, UploadFile 
from fastapi.responses import JSONResponse 
from PIL 
import Image 
import torch 
import torchvision.transforms as transforms 
import io 

app = FastAPI() 

# Load your model (same as step 1) 
model = MyModel() 
checkpoint = torch.load("path/to/your/checkpoint.pth",map_location=torch.device('cpu')) 
model.load_state_dict(checkpoint) model.eval() 

# Define a preprocessing pipeline for incoming images 
preprocess = transforms.Compose([ transforms.Resize((152, 152)), 
	  transforms.ToTensor(),
	  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) ]) 

@app.post("/predict/") 
async def predict(file: UploadFile = File(...)): 
	# Read the image 
	content = await file.read() 
	image = Image.open(io.BytesIO(content)).convert("RGB") 
	input_tensor = preprocess(image).unsqueeze(0) 
	
	# Add batch dimension 
	# Make the prediction with torch.no_grad(): 
	output = model(input_tensor) 
	prediction = torch.argmax(output, dim=1).item() # Example for classification 
	return JSONResponse(content={"prediction": prediction}) 
```

## 4 - Run the Server locally (Toy example)
Save the above script (e.g. server.py) and run it using FastAPI's `uvicorn`:

```bash uvicorn server:app --host 0.0.0.0 --port 8000 ``` 

This will start a server at `http://localhost:8000`. You can test it using tools like **curl** or Postman, or even write a Python client:

```python 
import requests 

url = "http://localhost:8000/predict/" 
files = {'file': open('path/to/test_image.jpg', 'rb')} 
response = requests.post(url, files=files) 
print(response.json()) # Output: {"prediction": ...} ```
```

## 4 Real - Deploy the Server

For a real-world service, deploy your server to platforms like AWS, Azure, or Google Cloud. You can use Docker to containerize the app, making deployment smoother. 



TODO: Word endpoint.
