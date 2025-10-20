---
tags:
  - Transformers
  - LLM
aliases: 
publish: true
slug: promt-engineering
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
The key is to specify different quality criteria and have the model evaluate the output from different dimensions on each iteration.
2. Context Aware Decomposition (CAD): Breaks down complex problems while maintaining awareness of the broader context.
```
I need to solve the following complex problem: [describe problem]
Please help me by:
1. Identifying the core components of this problem (minimum 3, maximum 5)
2. For each component:
	a. Explain why it's important to the overall problem
	b. Identify what information or approach is needed to address itc. Solve that specific component
3. After addressing each component separately, synthesize these partial solutions, explicitly addressing how they interact
4. Provide a holistic solution that maintains awareness of all the components and their relationships
    
Throughout this process, maintain a "thinking journal" that explains your reasoning at each step.
```
This approach has been extremely successful for solving complex programming challenges, and intricate analytical solutions.
3. Controlled Hallucination for Ideation (CHI): Harness hallucination for creative ideation.
```
I'm working on [specific creative project/problem]. I need fresh, innovative ideas that might not exist yet.

Please engage in what I call "controlled hallucination" by:

1. Generating 5-7 speculative innovations or approaches that COULD exist in this domain but may not currently exist 
2. For each one:a. Provide a detailed descriptionb. Explain the theoretical principles that would make it workc. Identify what would be needed to actually implement it
3. Clearly label each as "speculative" so I don't confuse them with existing solutions  
4. After presenting these ideas, critically analyze which ones might be most feasible to develop based on current technology and knowledge
The goal is to use your pattern-recognition capabilities to identify novel approaches at the edge of possibility.
```

4. Multi-Perspective Simulation (MPS): Leverage the model's ability to simulate different viewpoints, creating a more nuanced and comprehensive analysis.
```
I need a thorough analysis of [topic/issue/question].

Please create a multi-perspective simulation by:

1. Identifying 4-5 distinct, sophisticated perspectives on this issue (avoid simplified pro/con dichotomies)
    
2. For each perspective:a. Articulate its core assumptions and valuesb. Present its strongest arguments and evidencec. Identify its potential blind spots or weaknesses
    
3. Simulate a constructive dialogue between these perspectives, highlighting points of agreement, productive disagreement, and potential synthesis
    
4. Conclude with an integrated analysis that acknowledges the complexity revealed through this multi-perspective approach
    

Throughout this process, maintain intellectual charity to all perspectives while still engaging critically with each.
```
5. Calibrated Confidence Prompting (CCP)
```
I need information about [specific topic]. When responding, please:
1. For each claim or statement you make, assign an explicit confidence level using this scale:

- Virtually Certain (>95% confidence): Reserved for basic facts or principles with overwhelming evidence
- Highly Confident (80-95%): Strong evidence supports this, but some nuance or exceptions may exist
- Moderately Confident (60-80%): Good reasons to believe this, but significant uncertainty remains
- Speculative (40-60%): Reasonable conjecture based on available information, but highly uncertain
- Unknown/Cannot Determine: Insufficient information to make a judgment
    
2. For any "Virtually Certain" or "Highly Confident" claims, briefly mention the basis for this confidence
3. For "Moderately Confident" or "Speculative" claims, mention what additional information would help increase confidence    
4. Prioritize accurate confidence calibration over making definitive statements
    
This will help me appropriately weight your information in my decision-making.
```
This technique can be extremely powerful at preventing overconfident presentation of uncertain information.


### **Chain of Thought Prompts**

Chain of Thought (CoT) prompting encourages the model to break down complex reasoning into a series of intermediate steps, resulting in more thorough and well-structured final outputs.

A simple way to implement this is by having the LLM explicitly explain its reasoning in the response. For example, in a recent application using Gemini 2.5 with structured output, I added a `reasoning` field to the JSON. I forced the model to generate this field first, and instructed it to document its thought process step by step—breaking the problem into smaller, logical components before producing the final answer.

## Avoid imprecisiseness

```
Explain the concept prompt engineering. Keep the explanation short, only a few sentences, and don't be too descriptive.
```

better:

```
Use 2-3 sentences to explain the concept of prompt engineering to a high school student.
```




References:

https://ai.google.dev/gemini-api/docs/text-generation#best-practices

https://ai.google.dev/gemini-api/docs/files#prompt-guide

Gemini API https://ai.google.dev/api/generate-content#v1beta.GenerationConfig

JSON / Enum output https://ai.google.dev/gemini-api/docs/structured-output

about JSON Schemas, very interesting
* you can set a bunch of schema fields

```
{
  "type": enum (Type),
  "format": string,
  "description": string,
  "nullable": boolean,
  "enum": [
    string
  ],
  "maxItems": integer,
  "minItems": integer,
  "properties": {
    string: {
      object (Schema)
    },
    ...
  },
  "required": [
    string
  ],
  "propertyOrdering": [
    string
  ],
  "items": {
    object (Schema)
  }
}
```
Warning: When you're configuring a JSON schema, make sure to set propertyOrdering[], and when you provide examples, make sure that the property ordering in the examples matches the schema. When you're working with JSON schemas in the Gemini API, the order of properties is important. By default, the API orders properties alphabetically and does not preserve the order in which the properties are defined (although the [Google Gen AI SDKs](https://ai.google.dev/gemini-api/docs/sdks) may preserve this order). If you're providing examples to the model with a schema configured, and the property ordering of the examples is not consistent with the property ordering of the schema, the output could be rambling or unexpected.
- Support for JSON Schema is available as a preview using the field [`responseJsonSchema`](https://ai.google.dev/api/generate-content#FIELDS.response_json_schema) which accepts any JSON Schema with the following limitations:

- It only works with Gemini 2.5.
- While all JSON Schema properties can be passed, not all are supported. See the [documentation](https://ai.google.dev/api/generate-content#FIELDS.response_json_schema) for the field for more details.
- Recursive references can only be used as the value of a non-required object property.
- Recursive references are unrolled to a finite degree, based on the size of the schema.
- Schemas that contain `$ref` cannot contain any properties other than those starting with a `$`.

## Methodology

You can start with simple prompts and keep adding more elements and context as you aim for better results. Iterating your prompt along the way is vital for this reason. As you read the guide, you will see many examples where specificity, simplicity, and conciseness will often give you better results.


## Model zoo

https://ai.google.dev/gemini-api/docs/models



