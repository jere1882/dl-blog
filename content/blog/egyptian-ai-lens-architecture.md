---
has_been_reviewed: false
tag: Large Language Models
aliases:
publish: true
slug: egyptian-ai-lens-architecture
title: "Egyptian AI Lens: Architecture and Design of an LLM-Powered Art Analysis System"
description: Deep dive into the technical architecture, design decisions, and implementation details of the Egyptian AI Lens project, featuring Gemini API integration, structured outputs, and serverless deployment.
date: 2024-10-19
image: /sample-egyptian-images/VoK.jpg
---
## Introduction

 In 2023 I visited Egypt, and the highlight of my trip was by far the tombs in the Valley of the Kings. I revisited them twice during the same trip, mesmerized by the intricate paintings and carvings of gods and pharaohs, priests and commoners.

Back then, I remember thinking that these images presented fascinating challenges for applying computer vision. I would have loved to have an app capable of identifying the characters in each illustration, pointing out interesting details, and even live-translating hieroglyphs.

Fast-forward two years, and the advent of powerful multimodal foundation models like Gemini has made it fast and easy to prototype solutions for these kinds of tasks. 

In this blog post, I present **Egyptian AI Lens**, a web application that leverages Google’s Gemini model to analyze ancient Egyptian art. Users can upload images of tomb paintings, temple reliefs, or hieroglyphic inscriptions to receive detailed analyses, including character identification, historical context, and location insights.

![[Pasted image 20251109170928.png]]
*Demo: Identification of characters in a carving*

I designed this as a simple yet well-rounded project, encompassing model selection, performance evaluation, and full deployment via AWS Lambda and a web front end.

> 🏺 **Try it live**: [Egyptian AI Lens](https://www.jeremias-rodriguez.com/egyptian-ai-lens)

## System Architecture Overview

The Egyptian AI Lens follows a clean **frontend-backend separation** architecture, designed for scalability and maintainability:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Next.js       │    │   AWS Lambda     │    │   Google        │
│   Frontend      │◄──►│   + API Gateway  │◄──►│   Gemini API    │─► WWW
│   (TypeScript)  │    │   (Python/Docker)│    │   (Vision)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```
*Note: If grounding is enabled, Gemini is going to use Google search to ground its responses*

### Frontend: Next.js with TypeScript

The Next.js+TypeScript frontend has been kept simple on purpose, since the focus of this project is the backend. Its key role is to receive images uploaded by the users and desired settings; and then forward the request to the AWS Lambda ; and then present the response.
### Backend: AWS Lambda with Docker

The backend runs as a **Dockerized AWS Lambda function** with API Gateway integration, providing:

- **Serverless infrastructure** with automatic scaling
- **Docker container deployment** for consistent runtime environment
- **API Gateway integration** with request throttling and API key authentication
- **CloudWatch and Gemini monitoring** for logging and performance tracking

The Lambda code is written in Python, and it involves dynamically crafting a prompt where both the image and the user settings are injected, which is sent to Gemini.  More details below.

## Gemini API Integration

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

### Grounding

Grounding allows Gemini to browse the web in order to craft a better response, and it's one of the most powerful ways to augment the power of a foundation model. Gemini does not (yet) support both structured output and grounding at once ([check discussion](https://github.com/googleapis/python-genai/issues/665)), which is IMO a critical feature that should be available. 

In order to support grounding as an optional feature, I implemented a second strategy: instead of providing the response schema as a parameter, the desired response schema is injected in the prompt.

Grounding can be enabled using a checkbox in the UI.

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

### User settings

In order to demo how we can dynamically pick our best model settings, I added the following features:

* Dynamically pick the model (pro, flash or flash lite) based on the user willingness to wait for longer times
* Dynamically inject user-provided hints about the image location (tomb, temple, etc)
* Dynamically enable or disable grounding based on the user preferences

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

A Lambda was perfectly suited for my deployment needs:

1. This is a personal project, it will receive little traffic. Lambdas only run on demand.
2. I don't want to spend a lot of money, since this is not profitable. Lambdas are cheap and you pay only for what you use.
3. There is no need for heavy computing loads or long calculation times. Lambdas are only OK for small tasks like this.

I wrote a full step-by-step guide on how to deploy a small model in a Lambda [here](blog/deploying-your-model-in-aws-lambda).

## **Model Selection:**

I annotated a small dataset of 30 images, documenting the name of the most prominent characters that show on each image, as well as the period (old, medium or new kingdom).

Here is an example of the annotated set:

![[Pasted image 20251109223417.png]]
characters: Tutankhamun, Osiris
period: New Kingdom

I calculated stats on how good each model was at predicting the character names and the period across this test set. 
* The `period` prediction is a traditional multi-label classification problem, so we can use accuracy as a metric. 
* Comparing the list of identified `characters` is a little more complex, since the labels are variable-length lists of strings. Though I first tried to use straightforward string comparison to measure name matches, I settled for a more flexible **LLM as a Judge** approach. In this setup, `Ra-horakhty` is a match for `Ra` ; and even `Pharaoh` is considered a match for `Ramses II`.

| Variant                     | Period Accuracy | Character Partial Match | Character IoU |
| --------------------------- | --------------- | ----------------------- | ------------- |
| 2.5 Flash Lite              | 81.82%          | 63.64%                  |               |
| 2.5 Flash Lite w/ Grounding | 80.00%          | 40.00%                  |               |
| 2.5 Flash                   | 81.82%          | 86.00%                  |               |
| 2.5 Flash w/ Grounding      | 81.82%          | 54.55%                  |               |
| 2.5 Pro                     | 81.82%          | 86.00%                  |               |
| 2.5 Pro w/ Grounding        | 81.82%          | 63.64%                  |               |

Take these metrics with a pinch of salt, since they are only meant to give a super high level sense of how these models are performing. Most of the fields are not annotated.

The character match evaluation is particularly incomplete, since the stats are just comparing literal strings for equality (barring lowercase differences). 


Current system performance metrics:

| Model Speed | Avg Response Time | Lambda Cold Start | Cost per Request |
| ----------- | ----------------- | ----------------- | ---------------- |
| Regular     | 15-30s            | ~2-3s             | ~$0.02           |
| Fast        | 5-10s             | ~2-3s             | ~$0.008          |
| Super Fast  | 2-5s              | ~2-3s             | ~$0.003          |






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