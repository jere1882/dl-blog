---
tags:
  - Transformers
  - LLM
aliases:
publish: true
slug: promt-engineering
title: Working with LLMs in 2025
description: Notes from my experience working with LLMs
date: 2025-12-09
image: /thumbnails/pick_architecure.jpeg
---
# Working with LLMs in 2025

Oddly enough, I find myself entitling this blogpost *“Working with LLMs in 2025”*, because it’s so easy to see it as obsolete in a few months, thus the date tag becoming necessary.

The world of machine learning — and particularly large language models (LLMs) — is evolving at such a frantic pace that we can no longer predict what will happen in just a few short years. That said, in this post I will document the different practical lessons and concepts that I apply in my daily work as an ML engineer fully devoted to using LLMs for the last six months.

Most of my experience comes from the **Gemini family of models**, so examples will draw from them, though many of the principles generalize.

# The Gemini Model Family

Gemini offers three main variants: **Pro, Flash, and Flash Lite**. All of them can process text, images, and PDFs, but they differ in latency and capability.

- **Flash Lite** is the fastest for lightweight tasks (a few seconds for short text).  
- **Flash** strikes a good balance between speed and accuracy, making it the default choice for most workflows.  
- **Pro** is more powerful but noticeably slower, which makes it best suited for deeper analysis where latency is less critical.  

In practice, **latency matters**. For example, lightweight document parsing can be handled by Flash Lite in a couple of seconds, while the same task might take over 20 seconds with Pro. My general rule: *start with Flash, move up to Pro only if the extra reasoning is essential.*

| Model Variant                  | Key Capabilities / Differences                                                                                                                            | Context Window / Token Limits                          | Benchmark Strengths / Weaknesses                                                                            | Latency / Cost Trade-offs | Use-case Suggestions                                                   |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- | ------------------------- | ---------------------------------------------------------------------- |
| **Gemini 2.5 Pro**             | Top reasoning, multimodal understanding (text, image, video, audio), supports “thinking” (multi-step reasoning), function calling, search grounding, etc. | Up to **1M tokens** input, output in tens of thousands | Excels in math (AIME), long-context reasoning, coding, factual knowledge. Weakness: slower, more expensive. | Higher latency & cost     | Large docs, deep reasoning, multimodal analysis, error-sensitive tasks |
| **Gemini 2.5 Flash**           | Balanced speed vs accuracy, multimodal, good for structured outputs.                                                                                      | Similar input limit (~1M), smaller output              | Strong benchmarks across reasoning, code, and multimodal, slightly weaker than Pro                          | Lower latency, cheaper    | Default for production workloads                                       |
| **Gemini 2.5 Flash-Lite**      | Lowest latency/cost, retains core features                                                                                                                | Large input possible but shorter outputs               | Good for simple/mid tasks, trails on hardest reasoning/code/math                                            | Very low cost & latency   | Real-time apps, high-volume inference                                  |
| **Earlier Gemini (1.5, etc.)** | Smaller capacity, less efficient                                                                                                                          | Smaller token windows                                  | Weaker long-context and code reasoning                                                                      | Lower cost, faster        | Legacy pipelines, simple extraction                                    |
	My strategy: I pick gemini 2.5 flash by default. If it's smart enough, I try either adding thinking budget or moving up to pro.

# Gemini in the LLM Leaderboard

The following LLM leaderboards are interesting to find comparisons between SOTA models:
* https://www.vellum.ai/llm-leaderboard -> places gpt-5 / grok consistently first
* https://scale.com/leaderboard -> places either gpt-5 or gemini pro 2.5 consistenly at the top
* https://artificialanalysis.ai/models/comparisons/gpt-5-vs-gemini-2-5-pro

The recently released GPT-5 seems to be slightly better than Gemini 2.5 pro in most benchmarks. 

**Pricing**:

| Model / Mode                         | Input Price (per 1M tokens)                 | Output Price * (per 1M tokens) |
| ------------------------------------ | ------------------------------------------- | ------------------------------ |
| **Gemini 2.5 Pro — Standard**        | $1.25 (prompts ≤ 200k tokens)               | $10.00 (prompts ≤ 200k tokens) |
|                                      | $2.50 (prompts > 200k tokens)               | $15.00 (prompts > 200k tokens) |
| **Gemini 2.5 Flash — Standard**      | $0.30 (text/image/video) <br> $1.00 (audio) | $2.50 across modalities        |
| **Gemini 2.5 Flash-Lite — Standard** | $0.10 (text/image/video) <br> $0.30 (audio) | $0.40                          |
|                                      |                                             |                                |

**Comments**:
-  **Notice that the flash models are an 1+ order of magnitude cheaper than the pro version, and they can also be significantly faster.**
-  Input / pdf images in the prompt can be significantly slower than just text inputs.
- Caching of common prompt prefixes is automatically done, but in my experience it has no impact in latency, even if huge chunks of text are cached. 

# Latency analysis

I've worked in applications very sensible to latency, and it's important to understand what is driving latency. 

For a particular application, I collected hundreds of telemetry samples measuring the number of input tokens, output tokens and latency associated.  If we fit a simple linear regression model to this data:

duration_sec ~ candidate_tokens + prompt_tokens + constant

* The coefficient for the candidate tokens can be interpreted as the per-token decoding time -> **0.004 seconds = 4 milliseconds**
* The coefficient for the prompt tokens can be interpreted as the per-token encoding time. **0.000019 seconds = 0.019 milliseconds**
* The intercept gives the baseline overhead (network latency, model warmup, etc) **0.74 seconds**. 

These are super rough estimates, no to be taken literally, but it shows that the latency scales significantly up as the generated output is longer; whereas processing long inputs doesn't really add that much time.

Note: Gemini models 2.5 and above have the thinking mode feature, which gets enabled if you set a thinking token budget in the parameters. This can lead to a significant increase, and the default values are already high. Either restrict the budget or disable it.

![[correlation.png]]
## Structured Output

One of the most useful features is the ability to request **structured output** directly from the model. By providing a JSON schema, you can drastically improve consistency in the responses (in one case, schema adherence improved from ~4% errors to ~0.5%).

A couple of key takeaways:

- Don’t repeat the schema inside your prompt — the API’s schema field is enough.  
- Add a **reasoning field** or **chain of though field** as the first output item. Forcing the model to explain itself step by step increases reliability and makes debugging much easier.

## Controlling Hallucinations

Hallucinations are a fact of life when working with LLMs. Some approaches I’ve found effective include:

- **Prompt clarity**: Write prompts that leave little room for interpretation, and provide examples. Remove subjective terms by deterministic counterparts. E.g: "write a detailed summary of the following document" -> "write a detailed (500 word) summary of the following document" removes the ambiguity of what detailed means.
- **Single-task focus**: Avoid mixing too many tasks in a single request; multitasking increases the risk of spurious outputs.  The so called "curse of instructions" states that the per task and overall accuracy degrade as the number of instructions in the same prompt grows.
![[Pasted image 20250914122505.png]]
	This is one of the main challenges that I have found. LLMs really struggle to follow multi-criteria instructions, or slightly contradicting approaches. 
- **Monitoring at scale**: Don’t just test on isolated examples; use labeled datasets to evaluate performance systematically; and then monitor performance in production to detect subsets of inputs that may be troublesome, dataset shift, etc.
- **PII awareness**: Be deliberate about how personally identifiable information is handled. Consider blacking out or replacing sensitive information before sending information to an LLM.
- **Chain of thought or reasoning:** Make the model explicitly generate it before producing the actual result you are looking for. 
- **Add examples**: Provide I/O examples in your prompt to guide the response. 
