---
tags:
aliases: 
publish: false
slug: spark-notes
title: Notes on Spark
description: Assorted notes and takeaways from discussions
date: 2024-11-26
image: /thumbnails/backbones.png
---

# Apache Spark

Apache spark is a DISTRIBUTED data processsing framework designed for handling LARGE-SCALE datasets efficiently. It is particularly suited for ML/DL tasks on big data due to SCALABILITY and INTEGRATION W/ VARIOUS tools.

## When to use Spark?

- Large massive datasets that cannot fit into the memory of a single machine, e.g.a  dataset with terabytes or petabytes of data.
- Distributed workloads: Process or train models on a cluster of machines
- Integration with a broader big data pipeline e.g. Kafka
- Feature engineering at scale
- Spark can be used to distribute compute on CPUs if gpus are not available

## Why use Spark?
- Unified framework for data preparation, ML/DL, deployment
- Scalability over 100s or 1000s of nodes
- Speed
- Fault tolerance
- Integrated with python/pyspark, scala, java, etc

## Spark for ML
### Spark MLib for Traditional ML
The builtn library of spark for ML offers distributed algorithms for:
- classification, regression, clustering, collaborative filtering, PCA, precision/recall, etc.


## Concrete example

You have a cluster organized in a master-slave architecture.
- the master aka driver note is the entry point for Spark applications. it runs the Spark DRIVER program, which coordinates tasks and schedules execution.
- the worker nodes aka slaves execute tasks and store/process the data partitions.

A large dataset is split into small partitions and distributed across workers. sparks handles data partition automatically.

A developer would likely use Python on a local machine using jupyter notebooks or an IDE.

Spark applications connect to a cluster manager such as Apache YARN or Kubernetes.


A SparkSession is the entry point for Spark applications in Python (PySpark):

```python
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("ML Training Example") \
    .master("yarn") \  # or "k8s" or "local" for local mode
    .getOrCreate()
```

Load dataset from HDFS or local storage. the data is typically stored in a single, centralized location (e.g., HDFS, S3, or a local file system). When you load the data into Spark (step 2), Spark takes care of partitioning and distributing the data across the worker nodes in the cluster

```python
data = spark.read.csv("hdfs://path-to-large-dataset.csv", header=True, inferSchema=True)
```

Preprocess:

```python
# Select features and label columns
from pyspark.sql.functions import col

data = data.select(col("feature1").cast("float"), col("feature2").cast("float"), col("label").cast("int"))

# Split data into training and test sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
```
Train a model using spark MLib:

```python
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

# Assemble features into a single vector
assembler = VectorAssembler(inputCols=["feature1", "feature2"], outputCol="features")
train_data = assembler.transform(train_data)

# Train a logistic regression model
lr = LogisticRegression(featuresCol="features", labelCol="label")
model = lr.fit(train_data)

```

Evaluate:

```python
# Transform test data and make predictions
test_data = assembler.transform(test_data)
predictions = model.transform(test_data)

# Evaluate accuracy
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy: {accuracy}")
```

## Cluster SETUP

before running a Spark application, the cluster must already be set up and configured. The cluster configuration, which specifies which computers (nodes) are part of the cluster, is handled separately during the cluster setup phase. This setup includes defining how nodes communicate, how resources are allocated, and ensuring Spark is installed on each nod

1. Spark must be installed on all nodes (master and workers) in the cluster.
2. Cluster manager: A cluster manager (e.g., YARN, Kubernetes, Standalone, or Mesos) is used to coordinate the resources in the cluster. Configuration files (like spark-env.sh, spark-defaults.conf) specify details such as:
- Master node hostname/IP
- Worker node addresses
- Memory/CPU limits per node
3. Start spark services

on the master node: ./sbin/start-master.sh
on each worker: ./sbin/start-worker.sh spark://MASTER_HOSTNAME:7077 ; Replace MASTER_HOSTNAME with the actual address of the master node.

4. Access the Spark Web UI (default: http://MASTER_HOSTNAME:8080) to see connected worker nodes and their resource availability.

## Cluster manager example: using kubernetes if you have 100 local machines

1. Ensure all 100 computers are running the same OS e.g. ubuntu 20.04. All nodes must be on the same network and able to communicate with each other over private IPs. Assugn static IPs,
2. On all 100 nodes install doccker, kueadm, kubectl, kubelet via sudo apt install
3. Initialize kubernets master node on one computer:

```bash
sudo kubeadm init --pod-network-cidr=192.168.0.0/16
```

The `--pod-network-cidr` specifies the subnet for the pods. This must match the network plugin configuration (see Step 4).

Configure kubectl for the current user:

```bash
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
```

Test cluster status:

```
kubectl get nodes
```

A join comand must be saved to join worker nodes to the cluster:

```
kubeadm join <MASTER_IP>:6443 --token <TOKEN> --discovery-token-ca-cert-hash sha256:<HASH>
```

Insteall a pod network pluging e.g. calico and verify pods in the 
kube-system namespacae.

```
kubectl apply -f https://docs.projectcalico.org/manifests/calico.yaml
kubectl get pods -n kube-system
```

4. Join worker nodes to the cluster
Run kubeadm join saved earlier on the 99 workers:
```bash
sudo kubeadm join <MASTER_IP>:6443 --token <TOKEN> --discovery-token-ca-cert-hash sha256:<HASH>
```
After this, form the master, you should see all 100 nodes listed as ready if you run `kubectl get nodes`.

5. Configure kubernetes for spark
- create a namespace for spark `kubectl create namespace spark`
- configure persistent volumes where data is stored optional `kubectl apply -f pv.yaml`

6. Deploy spark on kubernetes
- build spark docker image `./bin/docker-image-tool.sh -r <your-registry> -t latest build`
- push the image to a container registry accesible to Kubernetes nodes:
```
docker tag spark:latest <your-registry>/spark:latest
docker push <your-registry>/spark:latest
```

submit a spark job
```
spark-submit \
  --master k8s://https://<KUBERNETES_MASTER_IP>:6443 \
  --deploy-mode cluster \
  --conf spark.kubernetes.container.image=<your-registry>/spark:latest \
  --conf spark.executor.instances=100 \
  local:///path/to/job.py
```
fter setting up the Kubernetes cluster and deploying Spark, you can indeed train models or run distributed data processing tasks using e.g. a jupyer notebook

## What if you don't have 100 machines and want to use a cloud provider?
You can use AWS (or other cloud providers) to set up a Kubernetes cluster and run Apache Spark. AWS makes it easy to manage large clusters through services like Amazon EKS (Elastic Kubernetes Service), and you can dynamically scale the number of virtual machines (EC2 instances) based on your workload.\

1. Prepare Your AWS Environment
- Create an AWS Account
- Set Up IAM Roles: Create an IAM role for EKS with permissions for Kubernetes and EC2 management. Attach these AWS managed policies:
- AmazonEKSClusterPolicy
- AmazonEKSWorkerNodePolicy
- AmazonEC2ContainerRegistryReadOnly
Optionally, use the AWS CLI to configure access credentials: `aws configure`

2. Set Up the Kubernetes Cluster with Amazon EKS
- Install AWS CLI, kubectl and eksctl in your local machine
- Create an EKS cluster:
```
eksctl create cluster \
  --name spark-cluster \
  --version 1.26 \
  --region us-west-2 \
  --nodegroup-name spark-nodes \
  --nodes 10 \
  --nodes-min 1 \
  --nodes-max 100 \
  --managed
```
Explained:
Set up a Kubernetes control plane (master nodes) managed by AWS.
Provision 10 worker nodes (EC2 instances).
Enable auto-scaling (1–100 nodes).
- Verify the cluster `kubectl get nodes`

3. Deploy Spark on the EKS Cluster
- Install Helm (a Kubernetes package manager)
- Install apache spark - This will deploy Spark master and worker pods across your EKS cluster.

```
helm install spark bitnami/spark \
  --namespace spark \
  --create-namespace
```
- Verify deployment `kubectl get pods -n spark`

4. Submit a spark job

```
spark-submit \
  --master k8s://https://<EKS_CLUSTER_API_ENDPOINT> \
  --deploy-mode cluster \
  --conf spark.kubernetes.container.image=bitnami/spark:latest \
  --conf spark.executor.instances=50 \
  local:///path/to/job.py
```
--master: Points to the Kubernetes API endpoint.
--executor.instances: The number of Spark executors to run.

The job specified in the spark-submit command is typically written as a Python script using the PySpark library. This Python script defines your data processing or machine learning task and leverages Spark’s distributed computing capabilities.

A common example is a word count job, which reads a text file, counts the occurrences of each word, and outputs the results.

```python
from pyspark.sql import SparkSession

# Initialize the Spark session
spark = SparkSession.builder.appName("WordCount").getOrCreate()

# Load the input text file from a distributed storage (e.g., S3, HDFS, etc.)
input_path = "s3://your-bucket/path/to/input.txt"  # Update with your file path
data = spark.read.text(input_path).rdd

# Process the data: Split lines into words
words = data.flatMap(lambda line: line[0].split(" "))

# Count each word
word_counts = words.map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)

# Save the result to a distributed storage
output_path = "s3://your-bucket/path/to/output"
word_counts.saveAsTextFile(output_path)

# Stop the Spark session
spark.stop()

```

or

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# Initialize Spark session
spark = SparkSession.builder.appName("MLPipeline").getOrCreate()

# Load data
data_path = "s3://your-bucket/path/to/data.csv"
data = spark.read.csv(data_path, header=True, inferSchema=True)

# Prepare features and labels
feature_columns = ["feature1", "feature2", "feature3"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data)

# Train/test split
train_data, test_data = data.randomSplit([0.8, 0.2])

# Train a regression model
lr = LinearRegression(featuresCol="features", labelCol="target")
lr_model = lr.fit(train_data)

# Evaluate on test data
predictions = lr_model.transform(test_data)
predictions.select("features", "target", "prediction").show()

# Stop Spark session
spark.stop()

```

or using libraries for DL
```python
import horovod.tensorflow as hvd
import tensorflow as tf
import pyspark

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors

# Horovod initialization
hvd.init()

# Load data using Spark
spark = SparkSession.builder.appName("DLTraining").getOrCreate()
data = spark.read.parquet("s3://your-bucket/path/to/preprocessed_data")

# Convert Spark DataFrame to NumPy arrays
features = data.select("features").rdd.map(lambda row: row[0].toArray()).collect()
labels = data.select("label").rdd.map(lambda row: row[0]).collect()

# Split data
train_features, test_features = features[:8000], features[8000:]
train_labels, test_labels = labels[:8000], labels[8000:]

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Horovod optimizer
optimizer = tf.keras.optimizers.Adam(0.001 * hvd.size())
optimizer = hvd.DistributedOptimizer(optimizer)

# Compile the model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Set up Horovod callbacks
callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback()
]

# Train the model
model.fit(
    train_features, train_labels,
    batch_size=128,
    epochs=10,
    callbacks=callbacks,
    validation_data=(test_features, test_labels)
)

# Save the model
if hvd.rank() == 0:  # Only save on rank 0
    model.save("s3://your-bucket/path/to/model")
```

# Apache Kafka

Apache Kafka is a distributed data streaming platform that is widely used for building real-time data pipelines and streaming applications. It acts as a messaging system that handles the ingestion, processing, storage, and distribution of data streams efficiently.

- Kafka runs as a cluster on multiple servers, offering high fault tolerance and scalability.

- Kafka enables low-latency data processing, making it suitable for applications that require real-time insights.
- high thorughput

A producer sends data to Kafka topics
A consumer reads data from Kafca topics
A topic is a named stream of data where producers write messages and consuers consume them
Each topic is divided into partitions enabling distributed storage and parallel processing.
A broker is a Kafka server responsible for storing and serving data

Key features:
- Scalable: Horizontal scalability through partitioning.
- Reliable: Built-in replication ensures data availability.
- Fast: High throughput and low latency.
- Versatile: Supports real-time and batch data pipelines.
- Open Source: Actively developed and widely adopted.

Apache Kafka, often used in conjunction with Apache Spark, is a powerful combination for real-time data processing and machine learning (ML) or deep learning (DL) projects. These tools excel in scenarios involving large-scale, real-time data streams. Here are examples of ML/DL projects where Kafka and Spark together would be particularly valuable:
- Real time fraud detection. Kafka would ingest transaction data streams for various sources.
- Personalized recommnedations in real time. Kafka'd be used to stream user activity data (clics, etc) in real time and spark would aggregate and preprocess this data at scale.