---
tag: Machine Learning
aliases: 
publish: true
slug: egyptian-ai-lens-architecture
title: "Egyptian AI Lens: Architecture and Design of an LLM-Powered Art Analysis System"
description: Deep dive into the technical architecture, design decisions, and implementation details of the Egyptian AI Lens project, featuring Gemini API integration, structured outputs, and serverless deployment.
date: 2024-12-15
image: /sample-egyptian-images/VoK.jpg
---

## Introduction

The **Egyptian AI Lens** is a web application that leverages Google's Gemini vision model to analyze ancient Egyptian art. Users can upload images of tomb paintings, temple reliefs, or hieroglyphic inscriptions to receive detailed analysis including character identification, historical context, and location insights.

This blog post provides a comprehensive technical overview of the system's architecture, design decisions, and implementation details. From frontend-backend separation to advanced prompt engineering with structured outputs, we'll explore how modern LLM APIs can be effectively integrated into production web applications.

> 🏺 **Try it live**: [Egyptian AI Lens](https://www.jeremias-rodriguez.com/egyptian-ai-lens)

## System Architecture Overview

The Egyptian AI Lens follows a clean **frontend-backend separation** architecture, designed for scalability and maintainability:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Next.js       │    │   Vercel         │    │   Google        │
│   Frontend      │◄──►│   Python         │◄──►│   Gemini API    │
│   (TypeScript)  │    │   Serverless     │    │   (Vision)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Frontend: Next.js with TypeScript

The frontend is built using **Next.js 14** with TypeScript, providing:

- **Server-side rendering** for optimal SEO and performance
- **Responsive UI components** built with Tailwind CSS
- **File upload handling** with drag-and-drop support
- **Real-time progress tracking** during analysis
- **Dark/light mode compatibility**

Key frontend features include:
- **Sample image gallery** with vacation photos from Egypt
- **Customizable analysis settings** (model speed, image type hints)
- **Structured result display** with detailed character information
- **Error handling** with comprehensive debugging information

### Backend: Python on Vercel Serverless

The backend runs as a **Vercel Python serverless function**, offering:

- **Zero-cost hosting** for low-traffic personal projects
- **Automatic scaling** based on demand
- **Fast cold start times** (~500ms)
- **Integrated deployment** with the frontend

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

A critical innovation in this project is the use of **Pydantic schemas** to constrain Gemini's responses:

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

### Current Architecture: Vercel Integration

The current deployment leverages **Vercel's integrated Python support**:

**Advantages:**
- **Zero configuration** deployment from Git
- **Automatic HTTPS** and CDN distribution
- **Built-in monitoring** and analytics
- **Free tier** suitable for personal projects
- **Seamless frontend-backend integration**

**Local Development Setup:**
```bash
# Frontend and API routes
npm run dev

# Python functions work automatically
# No separate backend server needed
```

### Performance Characteristics

Current system performance metrics:

| Model Speed | Avg Response Time | Cold Start | Cost per Request |
|-------------|-------------------|------------|------------------|
| Regular     | 15-30s           | +500ms     | ~$0.02          |
| Fast        | 5-10s            | +500ms     | ~$0.008         |  
| Super Fast  | 2-5s             | +500ms     | ~$0.003         |

## Next Steps: Migration to AWS Lambda

While Vercel provides excellent developer experience for prototyping, **AWS Lambda** offers advantages for production scaling:

### Proposed AWS Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Vercel        │    │   AWS Lambda     │    │   Google        │
│   Frontend      │◄──►│   Python         │◄──►│   Gemini API    │
│   (Static)      │    │   + API Gateway  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Benefits of AWS Lambda Migration

1. **Better Performance Control**:
   - **Configurable memory/CPU** allocation  
   - **Provisioned concurrency** to eliminate cold starts
   - **VPC integration** for enhanced security

2. **Advanced Monitoring**:
   - **CloudWatch Logs** for detailed debugging
   - **X-Ray tracing** for performance analysis
   - **Custom metrics** and alerting

3. **Cost Optimization**:
   - **Pay-per-millisecond** billing
   - **Reserved capacity** for predictable workloads
   - **Multi-region deployment** for global users

4. **Enhanced Scalability**:
   - **Higher timeout limits** (15 minutes vs 10 seconds)
   - **Larger payload sizes** for high-resolution images
   - **Concurrent execution** scaling

### Migration Strategy

**Phase 1: Infrastructure Setup**
```bash
# Terraform/CDK infrastructure
aws lambda create-function \
  --function-name egyptian-ai-lens \
  --runtime python3.11 \
  --memory-size 1024 \
  --timeout 60
```

**Phase 2: Code Adaptation**
- **Environment variable migration** for API keys
- **Response format standardization** 
- **Error handling enhancement** for AWS-specific errors
- **Logging integration** with CloudWatch

**Phase 3: Performance Optimization**
- **Container images** for faster startup times
- **Connection pooling** for Gemini API calls
- **Response caching** for repeated analysis requests

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
- **Structured outputs** to reduce hallucinations  
- **Dynamic model selection** for user control
- **Context injection** for improved accuracy
- **Robust error handling** for production reliability

The current **Vercel-based architecture** provides excellent developer experience and zero-cost hosting for personal projects. The planned **AWS Lambda migration** will unlock enhanced performance, monitoring, and scalability for production use.

> **Try the system**: [Egyptian AI Lens](https://www.jeremias-rodriguez.com/egyptian-ai-lens)  
> **Source code**: Available in the [blog repository](https://github.com/jeremias-rodriguez/dl-blog)

The intersection of **computer vision, large language models, and archaeology** opens fascinating possibilities for making historical knowledge more accessible. This project serves as a blueprint for similar applications across other domains. 