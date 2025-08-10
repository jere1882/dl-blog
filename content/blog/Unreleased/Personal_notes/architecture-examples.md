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
# ML Inference Service Cloud Architecture

Task: Run ML-inference during mission post-processing for room type recognition.

Design goals:
- Internal facing (public cannot directly call it)
- Cost effective, scalable
- Callable from various contexts 
- Runs a room-type recognition model
- Allows A/B testing
- Have a clear invocation+return payload schema defined through code in protobuf spec
![Pasted image 20241220133121](/assets/Pasted%20image%2020241220133121.png)Design:
- ML inference service provides a REST API service that IoT cloud components can use to provide ML features to our customers.
- Service also collects metrics that can be combined with customer feedback and robot data for creating KPI dashboards
- A REST API service that takes input data serialized as Protocol Buffer and returns inference result as a json
- Service doesn't need to have 100% uptime (the overall system sem2umf can run without it)
- Service is managed using CI/CD infrastructure
	- CI flow includes unit testing and basic integration testing
	- CD flow includes deployment management, integration/end to end testing

Notes:
- (AWS Lambda deployment) If lambda function is being used, loading model in memory for running inference is expensive so should only be done during lambda initialization
see OpenAPI specification in doc.

## Authentication
**API authentication** for a service that requires **AWS Signature Version 4 (SigV4)** to authenticate requests securely.

### Why AWS SigV4?

- **AWS SigV4** is considered the most secure way to authenticate API requests in AWS-based environments, as it ensures the identity and integrity of the requester, using AWS credentials that are signed in a way that is difficult to forge.
- The challenge is how to **integrate this authentication mechanism** into a system that may be constrained in terms of resources and requires lightweight solutions for signing requests.
