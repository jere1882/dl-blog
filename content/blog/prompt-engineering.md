---
tags:
  - Transformers
  - LLM
aliases: 
publish: true
slug: prompt-engineering
title: Prompt Engineering Notes
description: Notes from my experience working with LLMs
date: 2025-06-06
image: /thumbnails/pick_architecure.jpeg
---
The day-to-day job of many machine learning engineers and data scientists has changed dramatically. Instead of training and deploying costly models in the house, the most straightforward and often the best performing option is to send a request to e.g. Gemini, and have it solve your particular task for you.

As a machine learning engineer, I have seen much of my work reduced to picking the most suitable model and settings for my particular use case; crafting an appropriate prompt (which may include text and images), and processing the response. This setup may not be applicable to all cases, but when it is, it allows for much faster development than several years ago.

Prompt engineering has emerged as an umbrella term for many techniques, good practices and tricks that we use to nudge LLMs towards the desired behavior. LLMs are black boxes that are extremely hard to interpret, and may at times exhibit extremely arbitrary or unpredictable results such as hallucinations. 

# Is prompt engineering obsolete?

The AI world evolves at a frantic pace, and I've read interesting articles about prompt engineering as a discipline or job position being already obsolete.

The argument made is that as of 2025, the latest versions of LLM models are so smart that a very specialized expert is no longer necessary. These models are supposed to be able to solve tasks handling poorly written prompts, ranging from typing errors to inconsistencies in the request.

While I tend to agree with this opinion, I do consider that a non-trivial degree of familiarity with good practices, tricks and techniques are still necessary for many applications that involve prompting an LLM. In other words, we are not yet at a stage where prompt engineering can be so easily disregarded. 

Perhaps soon. At this point, gemini 2.5 shows dramatic improvements in accuracy for many of my projects when I iterate and refine my prompts, and it can be extremely sensitive to formatting and wording.
## Prompt design fundamentals

Generative language models work like an advanced auto completion tool. When you provide partial content, the model can provide the rest of the content or what it thinks is a continuation of that content as a response. When doing so, if you include any examples or context, the model can take those examples or context into account.


- **Specific instructions:** Craft clear and concise instructions that leave minimal room for misinterpretation. This may be obvious but it's worth repeating: Prompts have the most success when they are clear and detailed. If you have a specific output in mind, it's better to include that requirement in the prompt to ensure you get the output you want.
- **Add examples**: Use realistic few-shot examples to illustrate what you want to achieve. For basic text-generation, a zero-shot prompt may suffice without needing examples.it is essential to ensure a consistent format across all examples, especially paying attention to XML tags, white spaces, newlines, and example splitters.
- **Break it down step by step**: Divide complex tasks into manageable sub-goals, guiding the model through the process.
- **Specify output format**: Could be e.g. json via the API; of it it's just text you may ask for bulleted points, tables, etc.
- Add context
- **Review reasoning**: When you're not getting your expected response from the thinking models, it can help to carefully analyze Gemini's reasoning process. You can see how it broke down the task and arrived at its conclusion, and use that information to correct towards the right results.

## System instructions
You can guide the behavior of Gemini models with system instructions.
```
  config := &genai.GenerateContentConfig{
      SystemInstruction: genai.NewContentFromText("You are a cat. Your name is Neko.", genai.RoleUser),
  }
```

## Dealing with hallucinations

* Try dialing down the temperature setting or asking the model for shorter descriptions so that it's less likely to extrapolate additional details.
*
## LLM Settings

check the post for LLM settings
# Prompt engineering techniques
1. Recursive Self-Improvement Prompting (RSIP) : Iteratively improve the model output by using its own criticism.

```
I need you to help me create [specific content]. Follow this process:
1. Generate an initial version of [content]
2. Critically evaluate your own output, identifying at least 3 specific weaknesses
3. Create an improved version addressing those weaknesses
4. Repeat steps 2-3 two more times, with each iteration focusing on different aspects for improvement
5. Present your final, most refined version
    
For your evaluation, consider these dimensions: [list specific quality criteria relevant to your task]
```