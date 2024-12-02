---
tags:
  - Computer
  - Vision
aliases: 
publish: false
slug: a-review-of-computer-vision-backbones
title: A review of backbones used in computer vision
description: In this article I survey different backbone alternatives used in modern computer vision architectures
date: 2024-07-20
image: /thumbnails/backbones.png
---
# Overview of key concepts
1. **Scalability**: Understand how to design systems that can handle increased load by scaling horizontally (adding more machines) or vertically (upgrading machines). 
2. **Database Design**: Be clear on when to use SQL vs. NoSQL, sharding, replication, and indexing to optimize performance. 
3. **Caching**: Know how and where to use caching (e.g., Redis, Memcached) to reduce latency and offload databases. 
4. **Load Balancing**: Grasp concepts like round-robin, least connections, and how to use load balancers for fault tolerance and traffic distribution.
5. **Fault Tolerance and Consistency**: Learn strategies for ensuring high availability, data consistency (CAP theorem), and handling failures gracefully.

# Types of Servers: The pieces on the diagram

"server" refers to a physical or virtual machine that provides a specific type of service in a distributed system.
* **Application server**: Host and execute the core business logic of an application. They process requests from clients (e.g., web browsers) and interact with databases or other services ; e.g. Backend servers.
* **Web server**: Server static content (e.g. HTML, CSS), and sometimes act as the gateway to application servers, forwarding requests.
* **Load balancers**: Distribute incoming traffic across multiple servers to ensure high availability, scalability, and fault tolerance.
* **Database servers**: Store and manage data for the application. They respond to queries for retrieving, updating, or deleting data.
	* Relational databases: MySQL, PostgreSQL (structured, SQL-based).
	* NoSQL databases: MongoDB, Cassandra (unstructured or semi-structured).
* **Caching servers**: Store frequently accessed data in memory to reduce database load and improve performance. Example: Redis, Memcached.
* **Message Queue Servers**: Handle asynchronous communication between different components of a system. Decouple services by using queues for task processing or event streaming. E.g. Apache Kafka. 
* **DNS Servers** Translate human-readable domain names (e.g., example.com) into IP addresses.
* **File/Storage Servers**: Provide file storage and management
	* Object Storage: For unstructured data like images or logs each with metadata and a unique identifier, provided by AWS S3 buckets ; Google Cloud Storage. Designed for scalability. Built-in replication for durability and high availability.  e.g. Video streaming platforms store videos in S3 buckets and use databases to index file locations, metadata, and user data.
* **CDN** (Content Delivery Network): Cache and deliver static content (e.g., images, videos) from servers located close to users. Reduce latency and server load for global users. Files are uploaded to a central storage (e.g., object storage) ; CDNs replicate and cache files across edge servers worldwide.
# Databases

## Eventual consistency

Eventual Consistency: A consistency model used in distributed databases where the system ensures that, after some time and in the absence of further updates, all replicas of a piece of data will converge to the same value. It sacrifices immediate consistency for better performance, availability, and partition tolerance (as described in the CAP theorem).

Here's how it works: 
1. **Write to a Replica**: When data is updated, the write operation is applied to one or more replicas, but not necessarily all of them immediately. 
2. **Replication**: The updated data is asynchronously propagated to other replicas in the background. 
3. **Temporary Inconsistency**: During the propagation process, some replicas may hold outdated or inconsistent data. This is expected and acceptable under the eventual consistency model. 
4. **Convergence**: Over time, replication mechanisms (like anti-entropy or gossip protocols) ensure that all replicas receive the update and converge to the same value.
5. **Read Behavior**: - Some databases offer **read-your-writes consistency**: A client that performed a write can read its updates immediately from the replica it wrote to. - Other clients may see stale data until the system becomes consistent. 
Exaples: NoSQL databases such as Cassandra and DynamoDB ; CDNs ; DNS

## Horizontal scaling of SQL databases
Scaling SQL databases horizontally, also known as **horizontal scaling** or **sharding**, involves distributing the database across multiple servers to handle larger loads.

Data is split across multiple servers (shards), where each server holds a subset of the data. Each shard operates independently. Data is usually distributed based on a key (e.g., user ID or hash of a primary key). This ensures that data is evenly spread across servers.

Challenges:
* Joins and aggregations across shards become more complex
* Keeping replicas (duplicates) for redundancy and scalability, usually a single master for writes and N slaves for reads.

Horizontal scaling in SQL databases is harder than in NoSQL databases because SQL systems are traditionally built for strong consistency, which becomes difficult to maintain as the system grows. However, tools and techniques like partitioning, replication, and distributed SQL systems make horizontal scaling achievable for modern SQL databases.

## NoSQL databases

* key-value design
* High write/read throughput