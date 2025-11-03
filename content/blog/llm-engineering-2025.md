---
tags:
  - Transformers
  - LLM
aliases:
publish: true
slug: llm-engineering-2025
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

# API Parameters and Configuration

LLM Parameters are simple and yet their power is often overlooked. In my personal experience, when running LLM solutions at scale (e.g. thousands or millions of requests), bugs will emerge. Often obscure or unreasonable answers, formatting errors in the LLM response, etc. Often these are reactions to rare inputs, and sometimes it is impossible to reproduce or understand what caused the bug. LLMs are ultimately black boxes.

That said, the APIs often provide useful settings to constrain the LLM output. Specifying output format, bounds to the length of certain part(s) of the solution, or forcing certain instructions of the prompt to be given priority can be game-changing decisions.

## Sampling Parameters

### Temperature

*Short version*: this controls the randomness of the model output.

*Long version:* LLMs generate one token at a time in an auto-regressive way. The next token is sampled from a probability distribution. A model with temperature=0 will always pick the most probable token from that distribution. Large temperature values will encourage more diverse sampling on this distribution, increasing the weight of other possible tokens. If you are doing document parsing, you'd use temp=0 to encourage determinism. If you are generating creative content, like a song, you'd use large temperature.

* The model can still be non-deterministic with a temperature of 0, because tokens tied as the most probable can still be randomly picked.

| Temperature Range       | Typical Use Cases                                                                                                            | Summary                                       |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| Low (0.0 – 0.3)        | Technical Writing, Legal Documents, Customer Service Scripts, Factual Question Answering, Code Generation, Structured Data Extraction, Task-Specific Instructions | High accuracy, low creativity; reliable, factual, precise outputs |
| Medium-Low (0.3 – 0.5) | General Conversation, Content Summarization, Report Generation                                                               | Balanced accuracy and slight creativity; suitable for summarization and standard text generation |
| Medium-High (0.6 – 0.8)| Creative Writing Assistance, Brainstorming Sessions, Generating Multiple Alternatives                                         | Increased creativity; useful for ideation and generating varied responses |
| High (0.7 – 1.0)       | Content Ideation, Creative Writing, Marketing Campaigns, Poetry Generation, Experimental Writing, Exploring Unconventional Ideas | Maximum creativity and randomness; best for novel, unique, or unexpected content |

### Top P (Nucleus Sampling)

If you use Top P it means that only the tokens comprising the `top_p` probability mass are considered for responses, so a low `top_p` value selects the most confident responses. This means that a high `top_p` value will enable the model to look at more possible words, including less likely ones, leading to more diverse outputs. The general recommendation is to alter temperature or Top P but not both.

* Often a value between 0 and 1; where values closer to 0 restricting the AI to only the most likely words.
* Only tokens in the nucleus are considered for sampling, hence the name
* Lower Top P encourages the use of familiar, repetitive structure. For tasks that demand precision, such as drafting technical manuals, legal contracts or regulatory documents, a lower Top P ensures the AI remains focused on producing text that is clear, consistent and adheres to specific terminology and standards.
* Higher P values allow more diverse vocabulary and unexpected word combinations, useful for poetry generation, story plotting, vivid descriptions.

**Example**: 
- Low P (0.3): "The sunset in the west, coloring the sky red." 
- High P (0.8): "The dying day painted the heavens in a riot of crimson and gold, a fleeting masterpiece on nature's canvas."

**Temperature and Top-P Together:**

Temperature adjusts the overall probability distribution, while Top-p refines the selection of tokens based on their cumulative probability.

Temperature scales the logits (raw scores) before they're converted to probabilities. Top-P then filters this adjusted distribution, considering only the most probable tokens.

Temperature affects the overall shape of the distribution, Top-P determines how much of that distribution is considered for sampling.

While there may be some benefit in using them both, the increase in tuning complexity is probably not worth it. It is often advised to use either but not both. It is important to understand how they interact.

### Top K

Considers only the top K most probable tokens when generating text.

### Frequency Penalty and Presence Penalty

**Frequency penalty**: Reduces the likelihood of repeating the same words or phrases. Applies a penalty on the next token proportional to how many times that token already appeared in the response and prompt. The higher the penalty, the less likely a word will appear again.

**Presence penalty**: Encourages the model to introduce new topics.

## Truncation Parameters

### Max Tokens

Sets the maximum length of the generated text.

Important: The `max_tokens` parameter in the Gemini API primarily truncates the generated output when it reaches the specified token limit. It doesn't inherently bias the model towards generating shorter answers in all cases. If the model generates a response that exceeds the limit, it will be truncated.

A token is about 4 characters. 100 tokens ≈ 60-80 words.

### Stop Sequences

A string that stops the model from generating more tokens when encountered.

## Structured Output

One of the most useful features is the ability to request **structured output** directly from the model. By providing a JSON schema, you can drastically improve consistency in the responses (in one case, schema adherence improved from ~4% errors to ~0.5%).

A couple of key takeaways:

- Don’t repeat the schema inside your prompt — the API’s schema field is enough.  
- Add a **reasoning field** or **chain of though field** as the first output item. Forcing the model to explain itself step by step increases reliability and makes debugging much easier.

### Structured Output in Python

At first, I struggled a little bit to understand how to apply structured output in Python. This is because the official documentation provides examples using Pydantic models, for instance the following Pydantic schema specifies the field names and types for an application that attempts to extract details from an input photo of Egyptian art.

```python
class EgyptianArtAnalysis(BaseModel):
	"""Analysis results for Egyptian art and hieroglyphs.
	Field order is preserved by the Google Gen AI SDK when building the schema.
	"""
		picture_location: str = Field(
			description="Your best guess as to where this picture could have been taken - specific Valley of the Kings tomb, temple wall, etc. Use speculative language unless very confident, and justify your guess."
		)
		
		date: str = Field(
			description="Your best guess as to when this may have been produced. Give one of the major Egyptian periods like Old Kingdom, Middle Kingdom, or New Kingdom."
		)
	(...)
```

This may be good enough for certain use cases, but Gemini's structured output allows for a much richer set of constraints to be imposed in the output:
* **maxItems**, **minItems**: the length of sequence fields
* **required**: which of the fields in the schema are required and cannot be omitted
* **propertyOrdering**: the order for generating each field. Beware that in many languages, Gemini will use alphabetic ordering if you don't/. specify this. The reasoning should always go first, to force the model to reason before outputing a final answer!
* **type: string + enum: ['a', 'b', 'c']**: This type of declaration allows for a set of enumerate values to be the only possible choices for a given field.
* **minLength, maxLength**: min/max length for string fields ; similar **minimum/maximum** for integers
Check out the full list in the [official docs](https://ai.google.dev/api/caching#Schema). 

If you wish to use these features, then use the schema as follows:

```python
EGYPTIAN_ART_ANALYSIS_SCHEMA = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        "characters": genai.protos.Schema(
            type=genai.protos.Type.ARRAY,
            items=CHARACTER_SCHEMA,
            description="List of characters/deities/people identified in the scene. Empty if no clear characters depicted. Max 5 characters.",
            min_items=0,
            max_items=5
        ),
        "date_period": genai.protos.Schema(
            type=genai.protos.Type.STRING,
            enum=["old_kingdom", "middle_kingdom", "new_kingdom", "other"],
            description="The historical period when this artwork was produced. Choose one: old_kingdom, middle_kingdom, new_kingdom, or other."
        ),
    (...)
```

### Important Notes on JSON Schema Property Ordering

When working with JSON schemas in the Gemini API, the order of properties is important. By default, the API orders properties alphabetically and does not preserve the order in which the properties are defined (although the [Google Gen AI SDKs](https://ai.google.dev/gemini-api/docs/sdks) may preserve this order). 

**Warning**: When you're configuring a JSON schema, make sure to set `propertyOrdering[]`, and when you provide examples, make sure that the property ordering in the examples matches the schema. If you're providing examples to the model with a schema configured, and the property ordering of the examples is not consistent with the property ordering of the schema, the output could be rambling or unexpected.

### JSON Schema Support Details

Support for JSON Schema is available as a preview using the field [`responseJsonSchema`](https://ai.google.dev/api/generate-content#FIELDS.response_json_schema) which accepts any JSON Schema with the following limitations:

- It only works with Gemini 2.5.
- While all JSON Schema properties can be passed, not all are supported. See the [documentation](https://ai.google.dev/api/generate-content#FIELDS.response_json_schema) for the field for more details.
- Recursive references can only be used as the value of a non-required object property.
- Recursive references are unrolled to a finite degree, based on the size of the schema.
- Schemas that contain `$ref` cannot contain any properties other than those starting with a `$`.

The full JSON Schema specification includes properties like:
- `type`, `format`, `description`, `nullable`
- `enum` for enumerated values
- `maxItems`, `minItems` for array length constraints
- `properties` for nested objects
- `required` for mandatory fields
- `propertyOrdering` for controlling field generation order
- `items` for array item schemas

### Remarks
* Structured output allows you to have very fine control over the model output. This is great to control hallucination and to restrict the output dimensionality, potentially making the problem easier to solve.
* The model can still sometimes bypass these restrictions though. It is worth checking for output sanity in the client code, and monitor it.

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

## References

- [Gemini API Text Generation Best Practices](https://ai.google.dev/gemini-api/docs/text-generation#best-practices)
- [Gemini API Files and Prompt Guide](https://ai.google.dev/gemini-api/docs/files#prompt-guide)
- [Gemini API Generation Config](https://ai.google.dev/api/generate-content#v1beta.GenerationConfig)
- [Structured Output Documentation](https://ai.google.dev/gemini-api/docs/structured-output) 
