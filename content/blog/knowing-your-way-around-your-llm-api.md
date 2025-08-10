---
tags:
  - Transformers
  - LLM
aliases: 
publish: true
slug: llm-api-guide
title: Knowing Your Way Around Your LLM API
description: A practical guide to effectively using LLM APIs like Gemini in production
date: 2025-06-05
image: /thumbnails/pick_architecure.jpeg
---
LLMs are arguably one of the most powerful tool in a machine learning engineer's toolkit. When faced with requirements for an AI/ML solution, perhaps the first question an engineer needs to ask is whether a simple Gemini or Claude API request is enough to solve the problem.

The days where ML engineers needed to train and label their own datasets and train custom models are, in many cases, over. An increasingly large portion of problems can be solved with acceptable and even outstanding accuracy by general agents like Gemini 2.5. However, real-time or edge applications may still require custom solutions.

This post is about knowing our way around LLM APIs, with a focus on Gemini. LLM Parameters are simple and yet their power is often overlooked. In my personal experience, when running LLM solutions at scale (e.g. thousands or millions of requests), bugs will emerge. Often obscure or unreasonable answers, formatting errors in the LLM response, etc. Often these are reactions to rare inputs, and sometimes it is impossible to reproduce or understand what caused the bug. LLMs are ultimately black boxes.

That said, the APIs often provide useful settings to constrain the LLM output. Specifying output format, bounds to the length of certain part(s) of the solution, or forcing certain instructions of the prompt to be given priority can be game-changing decisions.

# Gemini API

## Sampling parameters

**Temperature**: 

*Short version*: this controls the randomness of the model output.
*Long version:* LLMs generate one token at a time in an auto-regressive way. The next token is sampled from a probability distribution. A model with temperature=0 will always pick the most probable token from that distribution. Large temperature values will encourage more diverse sampling on this distribution, increasing the weight of other possible tokens.  If you are doing document parsing, you'd use temp=0 to encourage determinism. If you are generating creative content, like a song, you'd use large temperature.

* The model can still be non-deterministic with a temperature of 0, because tokens tied as the most probable can still be randomly picked.

| Temperature Range       | Typical Use Cases                                                                                                            | Summary                                       |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| Low (0.0 – 0.3)        | Technical Writing, Legal Documents, Customer Service Scripts, Factual Question Answering, Code Generation, Structured Data Extraction, Task-Specific Instructions | High accuracy, low creativity; reliable, factual, precise outputs |
| Medium-Low (0.3 – 0.5) | General Conversation, Content Summarization, Report Generation                                                               | Balanced accuracy and slight creativity; suitable for summarization and standard text generation |
| Medium-High (0.6 – 0.8)| Creative Writing Assistance, Brainstorming Sessions, Generating Multiple Alternatives                                         | Increased creativity; useful for ideation and generating varied responses |
| High (0.7 – 1.0)       | Content Ideation, Creative Writing, Marketing Campaigns, Poetry Generation, Experimental Writing, Exploring Unconventional Ideas | Maximum creativity and randomness; best for novel, unique, or unexpected content |



**Top P** (aka Nucleus Sampling): If you use Top P it means that only the tokens comprising the `top_p` probability mass are considered for responses, so a low `top_p` value selects the most confident responses. This means that a high `top_p` value will enable the model to look at more possible words, including less likely ones, leading to more diverse outputs. The general recommendation is to alter temperature or Top P but not both.

* often a value between 0 and 1 ; where values closer to 0 restricting the AI to only the most likely words.
* only tokens in the nucleous are considered for sampling, hence the name
* lower Top P encourages the use of familiar, repetitive structure. for tasks that demand precision, such as drafting technical manuals, legal contracts or regulatory documents, a lower Top P ensures the AI remains focused on producing text that is clear, consistent and adheres to specific terminology and standards.
* higher P values allow more diverse vocabulary and unexpected word combinations, useful for poetry generation, story plotting, vivid descriptions.
	
Example: Low P (0.3): "The sunset in the west, coloring the sky red." 
High P (0.8): "The dying day painted the heavens in a riot of crimson and gold, a fleeting masterpiece on nature's canvas."

```
TEMPERATURE AND TOP-P TOGETHER
Temperature adjusts the overall probability distribution, while Top-p refines the selection of tokens based on their cumulative probability.

Temperature scales the logits (raw scores) before they're converted to probabilities. Top-P then filters this adjusted distribution, considering only the most probable tokens.

Temperature affects the overall shape of the distribution, Top-P determines how much of that distribution is considered for sampling. 

While there may be some benefit in using them both, the increse in tuning complexity is probably not worth it. It is often advised to use either but not both. It is important to understand how they interact.
```

**Top K**: Considers only the top K most probable tokens when generating text.

**Frequency penalty**: Reduces the likelihood of repeating the same words or phrases. Applies a penalty on the next token proportional to how many times that token already appeared in the response and prompt. The higher the penalty, the less likely a word will appear again.

**Presence penalty**: Encourages the model to introduce new topics.
## Truncation

**Max tokens**: Sets the maximum length of the generated text. 
- Important: The `max_tokens` parameter in the Gemini API primarily truncates the generated output when it reaches the specified token limit. It doesn't inherently bias the model towards generating shorter answers in all cases. If the model generates a response that exceeds the
- A token is about 4 chars.

**Stop sequences**: A string that stops the model from generating more tokens


I do not entirely agree with this flowchart, but I find it interesting at least as food for thought:
![Pasted image 20250614232415](/assets/Pasted%20image%2020250614232415.png)


* response type (text, json, enum) 
* max output tokens. a token is ~ 4 characters. 100 tokens =~ 60-80 words. 

# Default values



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

RAG
