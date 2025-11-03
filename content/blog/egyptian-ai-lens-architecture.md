---
tag: Machine Learning
aliases:
publish: true
slug: egyptian-ai-lens-architecture
title: "Egyptian AI Lens: Architecture and Design of an LLM-Powered Art Analysis System"
description: Deep dive into the technical architecture, design decisions, and implementation details of the Egyptian AI Lens project, featuring Gemini API integration, structured outputs, and serverless deployment.
date: 2024-10-19
image: /sample-egyptian-images/VoK.jpg
---

## Introduction

The **Egyptian AI Lens** is a web application that leverages Google's Gemini vision model to analyze ancient Egyptian art. Users can upload images of tomb paintings, temple reliefs, or hieroglyphic inscriptions to receive detailed analysis including character identification, historical context, and location insights.

This blog post provides a technical overview of the system's architecture, design decisions, and implementation details. From frontend-backend separation to advanced prompt engineering with structured outputs, we'll explore how modern LLM APIs can be swiftly integrated into production web applications. 

> 🏺 **Try it live**: [Egyptian AI Lens](https://www.jeremias-rodriguez.com/egyptian-ai-lens)

## System Architecture Overview

The Egyptian AI Lens follows a clean **frontend-backend separation** architecture, designed for scalability and maintainability:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Next.js       │    │   AWS Lambda     │    │   Google        │
│   Frontend      │◄──►│   + API Gateway  │◄──►│   Gemini API    │
│   (TypeScript)  │    │   (Python/Docker)│    │   (Vision)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Frontend: Next.js with TypeScript

The frontend is built using **Next.js 14** with TypeScript, providing:

- **Server-side rendering** for optimal SEO and performance
- **Responsive UI components** built with Tailwind CSS
- **File upload handling** with drag-and-drop support
- **Real-time progress tracking** during analysis
- **Dark/light mode compatibility**

Frontend has never been amongst my interests, and I do admit that the interface is as simple as possible while being functional. I am aware that there are security flaws.

### Backend: AWS Lambda with Docker

The backend runs as a **Dockerized AWS Lambda function** with API Gateway integration, providing:

- **Scalable serverless infrastructure** with automatic scaling
- **Docker container deployment** for consistent runtime environment
- **API Gateway integration** with request throttling and API key authentication
- **CloudWatch monitoring** for logging and performance tracking
- **Configurable memory and timeout** (currently 512MB, 30s timeout)
- **CORS support** for browser-based requests from the Next.js frontend

Below are more details as to why I chose these specific deployment and model options.

## Deep Dive: Gemini API Integration

### Model Selection Strategy

One of the key architectural decisions was implementing **dynamic model selection** based on user preferences:

```python
def get_model_by_speed(speed: str) -> str:
    model_mapping = {
        'regular': 'gemini-2.5-pro',        # Most thorough
        'fast': 'gemini-2.5-flash',         # Balanced (default)
        'super-fast': 'gemini-2.5-flash-lite'  # Fastest
    }
    return model_mapping.get(speed, 'gemini-2.5-flash')
```

This approach allows users to **trade-off between analysis quality and response time**:
- **Regular** (~15-30s): Most detailed analysis using the flagship model
- **Fast** (~5-10s): Balanced performance, recommended for most users  
- **Super Fast** (~2-5s): Quick analysis for instant feedback

### Prompt Engineering with Context Injection

The system uses **context-aware prompting** by injecting user-provided hints about the image type:

```python
def create_egyptian_art_prompt(image_type_hint: str) -> str:
    base_prompt = """Analyze this ancient Egyptian art image and provide detailed analysis..."""
    
    if image_type_hint != 'unknown':
        context_hints = {
            'tomb': "This appears to be from a tomb or burial site...",
            'temple': "This appears to be from a temple complex...", 
            'other': "This appears to be other Egyptian artwork..."
        }
        base_prompt += f"\n\nContext hint: {context_hints[image_type_hint]}"
    
    return base_prompt
```

This **contextual prompting** significantly improves accuracy by:
- **Focusing the model's attention** on relevant historical periods
- **Reducing hallucinations** through targeted context
- **Improving character identification** with location-specific knowledge

### Structured Output Implementation

A critical piece in this project is the use of **[parameter schemas](/blog/llm-engineering-2025#structured-output-in-python)** to constrain Gemini's responses. In my current company, which uses Gemini at scale, I reduced the operational error rate from 4% to under 0.5% by adopting structured output via input schema. It strongly constrains the model to respond in the appropriate model.

```python
from pydantic import BaseModel
from typing import List

class Character(BaseModel):
    character_name: str
    reasoning: str  
    description: str
    location: str

class EgyptianArtAnalysis(BaseModel):
    ancient_text_translation: str
    characters: List[Character]
    location_guess: str
    interesting_detail: str
    historical_date: str
```

This approach provides several key benefits:

1. **Consistent Response Format**: Eliminates parsing errors from inconsistent JSON
2. **Reduced Hallucinations**: Structured fields guide the model's responses
3. **Type Safety**: Automatic validation of response data types
4. **Better UX**: Predictable data structure enables rich frontend displays

### Retry Logic and Error Handling

Production systems require robust error handling. The Egyptian AI Lens implements **exponential backoff retry logic**:

```python
async def analyze_with_retry(image_data: str, max_retries: int = 2):
    for attempt in range(max_retries + 1):
        try:
            result = await call_gemini_api(image_data)
            return result
        except Exception as e:
            if attempt < max_retries and is_retryable_error(e):
                wait_time = (2 ** attempt) * 1.0  # Exponential backoff
                await asyncio.sleep(wait_time)
                continue
            raise e
```

This handles:
- **5xx server errors** from the Gemini API
- **Rate limiting** during high-traffic periods  
- **Temporary network issues**
- **Service unavailability**

## Hosting and Deployment

### Current Architecture: AWS Lambda with API Gateway

The current deployment leverages **AWS Lambda** with Docker containerization and **API Gateway** for request routing:

**Architecture Components:**

1. **Frontend (Vercel)**: Next.js static site hosted on Vercel
2. **API Gateway**: AWS API Gateway handles CORS, request throttling, and API key authentication
3. **Lambda Function**: Dockerized Python application running in AWS Lambda
4. **Google Gemini API**: External AI service for image analysis

**Deployment Process:**

The Lambda function is deployed using a Docker container:

```bash
# Build and push Docker image to ECR
docker build -t egyptian-art-analyzer .
docker tag egyptian-art-analyzer:latest <ecr-repo>:latest
docker push <ecr-repo>:latest

# Create/update Lambda function with container image
aws lambda create-function \
  --function-name egyptianArtAnalyzer \
  --package-type Image \
  --code ImageUri=<ecr-repo>:latest \
  --role <execution-role>
```

**Advantages of AWS Lambda Architecture:**

1. **Better Performance Control**:
   - **Configurable memory/CPU** allocation (512MB configured)
   - **Docker containerization** for consistent runtime environment
   - **Longer timeout limits** (30 seconds) for complex analysis

2. **Advanced Monitoring**:
   - **CloudWatch Logs** for detailed debugging and request tracing
   - **CloudWatch Metrics** for performance monitoring
   - **Error tracking** and alerting capabilities

3. **Security Features**:
   - **API Gateway API keys** for request authentication
   - **Usage plans** for rate limiting and quota management
   - **Environment variables** for secure API key storage
   - **CORS configuration** for browser security

4. **Scalability**:
   - **Automatic scaling** based on concurrent requests
   - **Pay-per-request** pricing model
   - **No infrastructure management** required

**Performance Characteristics:**

Current system performance metrics:

| Model Speed | Avg Response Time | Lambda Cold Start | Cost per Request |
|-------------|-------------------|-------------------|------------------|
| Regular     | 15-30s           | ~2-3s            | ~$0.02          |
| Fast        | 5-10s            | ~2-3s            | ~$0.008         |  
| Super Fast  | 2-5s             | ~2-3s            | ~$0.003         |

**Environment Configuration:**

- **Lambda Environment Variables**: `GOOGLE_API_KEY` stored securely in Lambda configuration
- **Frontend Environment Variables**: `NEXT_PUBLIC_LAMBDA_API_URL` and `NEXT_PUBLIC_API_KEY` for API Gateway endpoint and authentication
- **API Gateway**: Configured with usage plans and throttling limits

## Technical Lessons Learned

### 1. Structured Outputs Are Game-Changing

The single most impactful technical decision was implementing **Pydantic-enforced structured outputs**. This eliminated ~90% of parsing errors and dramatically improved response quality.

### 2. Context Injection Improves Accuracy

Allowing users to provide **image type hints** (tomb, temple, other) significantly improved model accuracy by focusing attention on relevant historical contexts.

### 3. Model Speed Options Enhance UX  

Offering **multiple speed tiers** provides users control over the speed-accuracy tradeoff, essential for interactive applications.

### 4. Error Handling is Critical

Robust **retry logic and error recovery** transforms a demo into a production-ready system. The Gemini API can be temperamental, making retry logic essential.

## Conclusion

The Egyptian AI Lens demonstrates how **modern LLM APIs can be effectively integrated** into production web applications. Key architectural decisions include:

- **Clean frontend-backend separation** for maintainability
- **AWS Lambda + API Gateway** for scalable serverless infrastructure
- **Docker containerization** for consistent runtime environments
- **Structured outputs** to reduce hallucinations  
- **Dynamic model selection** for user control
- **Context injection** for improved accuracy
- **Robust error handling** for production reliability
- **API Gateway authentication** with usage plans for security

The current **AWS Lambda architecture** provides production-ready infrastructure with:
- **Automatic scaling** and pay-per-use pricing
- **CloudWatch monitoring** for observability
- **API Gateway** for request management and security
- **Docker deployment** for consistent runtime environments

This architecture scales from personal projects to production workloads while maintaining cost efficiency and operational simplicity.

> **Try the system**: [Egyptian AI Lens](https://www.jeremias-rodriguez.com/egyptian-ai-lens)  
> **Source code**: Available in the [Lambda repository](https://github.com/jere1882/egyptianArtAnalyzer)

The intersection of **computer vision, large language models, and archaeology** opens fascinating possibilities for making historical knowledge more accessible. This project serves as a blueprint for similar applications across other domains. 