---
has_been_reviewed: false
tag: Large Language Models, Deep Learning Basics
tags:
  - large-language-models
aliases: 
publish: true
slug: llm-architectures
title: Modern LLM Architectures
description: A snapshot of LLM architectures in mid 2025
date: 2025-06-29
image: /thumbnails/pick_architecure.jpeg
---
As of **mid‑2025**, the LLM landscape has **consolidated** around **three dominant model families**, each led by a major player:

|Family|Organization|Key Strengths|Flagship Models|
|---|---|---|---|
|**GPT**|OpenAI|Scale, general performance, multimodal speed, ecosystem|GPT‑3 → GPT‑4 → GPT‑4o → GPT‑4.5|
|**Claude**|Anthropic|Safety, transparency, reasoning depth|Claude 1 → 2 → 3 → 3.5 → 3.7|
|**Gemini**|Google DeepMind|Multimodal context, tool integration, robotic/agent use|Gemini 1 → 1.5 → 2.5|
Let's have a look at each one.

# The GPT family

OpenAI’s **Generative Pre-trained Transformer (GPT)** models have redefined natural language processing and AI usability since the release of GPT-2 in 2019. Over successive iterations, the GPT family has introduced new capabilities, scaled in size and sophistication, and pioneered multimodal integration, tool use, and real-time AI assistance. These models form the backbone of **ChatGPT**, one of the most widely used AI platforms in the world today.

The GPT series has been at the **forefront of general-purpose AI**, shaping how people work, learn, and create. While **GPT-4 remains the gold standard for quality**, **GPT-3.5 Turbo** powers much of the world's AI infrastructure, and **GPT-4o** opens the door to real-time, multimodal interaction for everyone.

## Architecture and traits

All models share a **decoder-only transformer foundation**, although key architectural improvements and design shifts have marked each generation.

| Model           | Architecture                                                            | Key Traits                                                                                      | Use Case Highlights                               |
| --------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| **GPT-3.5**     | Dense decoder-only transformer. 175B parameters.                        | Fast, affordable, good general-purpose performance; no multimodal or advanced reasoning         | Chatbots, summarization, translation, basic apps  |
| **GPT-4**       | Likely Mixture-of-Experts (MoE) 1.75 TRILLION parameters (hypothetical) | Trillion-scale model, strong reasoning, coding, and instruction following; top benchmark scores | Complex reasoning, coding, research, premium apps |
| **GPT-4 Turbo** | Optimized GPT-4 variant (details undisclosed)                           | Nearly same capability as GPT-4, but faster and cheaper to run                                  | ChatGPT Plus, production APIs, scale deployment   |

The GPT-4 family (including GPT-4, GPT-4 Turbo, and GPT-3.5 Turbo) is **not open-source**. OpenAI has **not publicly disclosed**:
- The exact **parameter count** 
- The **training data sources**
- The **precise architecture** (e.g., confirmation of Mixture-of-Experts, layer counts, tokenizers, etc.)
![[Pasted image 20250629152728.png]]
OpenAI did publish a full technical paper on GPT-3 in 2020: *Language Models are Few-Shot Learners (Brown et al., 2020)*
## Training

### Training data

| Model     | What We Know (or Don’t)                                                                                                                                                                                   |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **GPT‑3** | Trained on ~300 billion tokens from public internet data — Common Crawl, WebText, books, Wikipedia, etc. Not open-source, but the paper listed categories.                                                |
| **GPT‑4** | **Unknown**. OpenAI has not disclosed sources, size, or preprocessing. Likely includes more curated, filtered, and possibly proprietary data (e.g., licensing deals with publishers or social platforms). |
🧠 **Insight**: The trend has shifted from “massive web scrape” (GPT-3) to **curated, diverse, high-quality datasets** (GPT-4+), including code, math, dialogues, images, and potentially structured documents.

### Training methodology

| Model           | Method                                                                                                                                                                                                                 | Notes                                                                                               |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **GPT‑3**       | **Next-token prediction (causal LM)** over a massive unsupervised corpus                                                                                                                                               | Standard autoregressive transformer; purely self-supervised pretraining                             |
| **GPT‑4**       | Presumed same base (next-token prediction), but with **enhancements**:  <br />• Possibly **Mixture-of-Experts (MoE)**  <br />• Reinforcement Learning from Human Feedback (**RLHF**)  <br />• Fine-tuning on custom datasets | OpenAI has not confirmed this, but it's consistent with performance and hints in public statements. |
| **GPT‑4 Turbo** | Unknown. Same general approach, likely with infrastructure optimizations (e.g., distillation, quantization, MoE routing improvements)                                                                                  | Possibly trained using custom hardware/software stack for efficiency                                |

*  With **RLHF**, the model is fine-tuned using rankings/preferences from humans to make it more helpful, safe, and aligned.
* Even if GPT-4 has not released training methodology, all known autoregressive LLMs, even in 2025, still rely on **next-token prediction as the base pretraining objective** — it's fundamental to how transformers learn. OpenAI’s own API docs and system behavior imply this. The models **predict the next token given a context**, which is literally what next-token prediction is. Features like _logprobs_, _top-k sampling_, and _greedy decoding_ all stem from a next-token likelihood model.
### Multimodal Capabilities

| Model                    | Modalities Supported      | Details                                                                                                                                   |
| ------------------------ | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **GPT‑3**                | **Text only**             | No native support for images or audio                                                                                                     |
| **GPT‑4**                | **Text + Images           | In its multimodal variant, GPT‑4 can process images (e.g., charts, screenshots, diagrams). This version powers tools like **Be My Eyes**. |
| **GPT‑4 Turbo / GPT‑4o** | **Text + Images + Audio** | GPT‑4o adds real-time voice and audio processing, full multimodal interaction (vision, speech, code)                                      |
🧠 **Insight**: GPT-3 was strictly text-only, but GPT-4 marks the **start of OpenAI’s serious move into multimodality** — though the base GPT-4 may still be text-only unless otherwise activated.