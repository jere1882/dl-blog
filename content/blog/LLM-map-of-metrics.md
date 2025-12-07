---
has_been_reviewed: false
tag: Large Language Models
tags:
  - Transformers
  - LLM
aliases: 
publish: false
slug: llm-map-of-metrics
title: Metrics used in LLM tasks
description: A comprehensive summary of metrics used in LLM-Related tasks
date: 2025-06-06
image: /thumbnails/pick_architecure.jpeg
---
# Task-level metrics - Overview

This table provides an overview, and the following sections have deep dives into each metric.

When we evaluate language models (e.g. machine translation, summarization, text generation), we need metrics to compare:
* What the model produces (called "hypothesis" or "candidate")
* What is the ground truth ("reference" or "golden standard")

| **LLM Task Type**                        | **Example of Task**                       | **Key Metrics**                                                                 |
| ---------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------- |
| **Language Modeling (LM)**               | Next word prediction                      | Perplexity                                                                      |
| **Text Generation (Open-ended)**         | Story or code generation                  | Human evaluation (e.g. Elo), BLEU, ROUGE, Diversity scores (Distinct-n), MAUVE  |
| **Machine Translation**                  | English to German translation             | BLEU, METEOR, chrF++, TER, COMET                                                |
| **Summarization**                        | Document to summary                       | ROUGE, BLEU, METEOR, BERTScore, SummaC                                          |
| **Question Answering (QA)**              | SQuAD-style answer extraction             | Exact Match (EM), F1 score, BLEU, ROUGE-L                                       |
| **Retrieval-Augmented Generation (RAG)** | Answer generation with document retrieval | Retrieval: Recall@k, MRR, nDCG<br/>Generation: BLEU, ROUGE, BERTScore           |
| **Information Retrieval / Ranking**      | Search result ranking                     | nDCG, MAP, MRR, Precision@k, Recall@k                                           |
| **Text Classification**                  | Sentiment analysis, spam detection        | Accuracy, Precision, Recall, F1 Score, ROC-AUC                                  |
| **Dialogue / Conversational Agents**     | Chatbots, virtual assistants              | Human evaluation (helpfulness, safety, coherence), BLEU, METEOR, USR, DialogRPT |
| **Code Generation**                      | Code completion (e.g. Python)             | Exact Match, BLEU, CodeBLEU, pass@k (functional correctness)                    |
| **Reasoning / Logic Tasks**              | Logical deduction, arithmetic reasoning   | Accuracy, Exact Match, Chain-of-Thought correctness                             |
| **Fact Verification**                    | True/False claim verification             | Accuracy, Precision, Recall, F1 Score                                           |


# PPL (Perplexity)
Measures how well a model predicts a sequence of words. Lower perplexity means better model performance.
* A language model is often evaluated on a sequence prediction task
* e.g. input "the cat sat on the" - ground truth "mat"
* say the model predicts a probability distribution over its vocabulary of size K, and q_i is the predicted probability of the i-th word in the vocab

In theoretical cross entropy, we compare two full distributions:

![[Pasted image 20250616231326.png]]
But in practice p is not known, all we know is that the next correct token at position j. Thus we define p to be 0 in all components except p_j=1

Then, the sum simplifies just to -log q(x_j)

If we calculate this over N data points (a dataset), we get:

![[Pasted image 20250616231526.png]]
We finally define:

Perplexity = e^(H(p,q))

Why the e? Let's explain that with an example.

Input: "the cat sat on the"
Target "mat"

Model A output:  0.4 mat  0.3 rug 0.2 floor 0.1 bed
Model B output:  0.2 mat 0.4 rug 0.3 floor 0.1 bed

Cross entropy (unit would be "nats" from natural logarithm)

Model A cross entropy: 0.916 nats
Model B cross entropy: 1.609 nats

😕 Who cares? What's a "nat" anyway?

Model A perplexity => 2.5
Model B perplexity => 5.0

 "**Model A behaves as if it's picking between 2.5 words on average — Model B behaves as if it's choosing from 5 words.**"
You now have a measure of the **effective 'uncertainty' or 'confusion'** of the model.

In other words, it's a way to report the "branching factor" which is relevant in text generation, but not that relevant in e.g. cat/dog classification.

* Cross entropy is used as a loss but it's hard to interpret as a metric
* perplexity: Clear, interpretable: "How many real options the model behaves like it sees?"
**Perplexity isn't measuring what the model outputs — it's measuring how confidently it _thinks_**.

Beware though - A model with **ultra-low perplexity** may always predict the _most likely_, common, boring word. Minimizing perplexity blindly may make the model too safe. Human-like text has natural entropy and uncertainty.

You don't just want "min perplexity" — you want **controlled entrop**

## Definition: n-gram
An **n-gram** is simply a **sequence of 'n' consecutive words** (or tokens) from a text.

Text: `"The cat sat on the mat"`

| n   | n-grams                                                                       |
| --- | ----------------------------------------------------------------------------- |
| 1   | **Unigrams:** `"The"`, `"cat"`, `"sat"`, `"on"`, `"the"`, `"mat"`             |
| 2   | **Bigrams:** `"The cat"`, `"cat sat"`, `"sat on"`, `"on the"`, `"the mat"`    |
| 3   | **Trigrams:** `"The cat sat"`, `"cat sat on"`, `"sat on the"`, `"on the mat"` |
| 4   | **4-grams:** `"The cat sat on"`, `"cat sat on the"`, `"sat on the mat"`       |
# BLEU (Bilingual Evaluation Understudy)

Use case: Machine translation, but also other text generation tasks.

It measures **n-gram overlap** between generated text and reference texts.

1. Break both hypothesis and reference into n-grams (1-gram, 2-gram, 3-gram, etc)
2. Count how many n-grams in the hypothesis appear in the reference
3. Apply precision: What fraction of the hypothesis n-grams were correct?
4. Apply brievety penalty: Punish very short hypotheses (e.g. just one correct word)

![[Pasted image 20250609200452.png]]

*Example:*

*Reference:  The cat is on the mat
Hypothesis: The cat the cat on the mat*

Let's do bleu with n=2 for simplicity

Unigrams (1-grams):
- **Reference:**  
	`["The", "cat", "is", "on", "the", "mat"]`
- **Hypothesis:**  
    `["The", "cat", "the", "cat", "on", "the", "mat"]`

#### **Count matching unigrams:**

|Word|Reference Count|Hypothesis Count|Count Clipped to Reference Max|
|---|---|---|---|
|The|2|3|2 (max in ref: 2)|
|cat|1|2|1 (max in ref: 1)|
|is|1|0|0|
|on|1|1|1|
|mat|1|1||
✅ **Total clipped matches = 2 + 1 + 0 + 1 + 1 = 5**  
✅ **Total hypothesis unigrams = 7**

Unigram precision is thus P_1 = 5/7 = 0.714

Bigrams (2-grams):

- **Reference bigrams:**  
    `["The cat", "cat is", "is on", "on the", "the mat"]`
    
- **Hypothesis bigrams:**  
    `["The cat", "cat the", "the cat", "cat on", "on the", "the mat"]`

| Bigram  | Reference Count | Hypothesis Count | Count Clipped |
| ------- | --------------- | ---------------- | ------------- |
| The cat | 1               | 2                | 1             |
| on the  | 1               | 1                | 1             |
| the mat | 1               | 1                | 1             |
| others  | -               | not in ref       | 0             |

P_2 = 3/6 = 0.5

BP = |hyp| - |ref| = 1
BLEU = BP * exp((1/2) P_1 + (1/2) P_2) = 0.59

## Intuition

> **"A good machine-generated sentence should contain many of the same words and word patterns (n-grams) as high-quality human references — and in roughly the right order and length."**

Downsides:
* It **doesn't care if you missed some words in the reference** — only if you said extra or wrong things => it focuses on **precision** and not **recall**
* **Cannot detect paraphrases** ("The feline sits on the rug" scores poorly).
* Cannot check **meaning** — only **word overlap**.

🔷 **BLEU**:

- Measures **precision** — _How many of your output's n-grams appear in the reference?_  
    _("Did the system output contain what it should?")_
    
🔶 **ROUGE**:

- Measures **recall** — _How many of the reference's n-grams appear in your output?_  
    _("Did the system miss anything the reference had?")_

#  ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Use case: Popular for summarization tasks, also translation and text generation. ROUGE is commonly used in **text summarization evaluation** — because in summaries, covering key points (recall) is more important than matching exact phrasing.

### Rouge-N (N-gram overlap)

compares n-gram recall between system output and reference.

ROUGE_N = (overlapping n-grams) / (n-grams in reference)

Example for ROUGE-1:

Reference: The cat sat on the mat
Hypothesis: The cat sat mat

Unigrams in Hypothesis: `{ the, cat, sat, mat}`
Unigrams in Reference : `{ the, cat, sat, on, the, mat}`

ROUGE-1 = 4/5 = 0.8

meaning: you covered 80% of the important words.

ROUGE-L (Longest Common Subsequence - LCS)

Measures the length of the longest common subsequence (not necessarily consecutive) between system output and reference.

This captures sentence-level structure, not just n-grams.

Example (ROUGE-L):

Ref: the cat sat on the mat
Hyp: the cat mat
LCS: The cat mat => length 3

ROUGE_L Recall = (LCS length) / (reference length) = 3/6 = 0.5

ROUGE-S (Skip-bigram)

- Counts pairs of words in order, but allows gaps (skips) between them
- 'the cat sat on the mat' has skip-bigrams like (the, sat) and (the,mat)
# METEOR
Computes similarity between generated (candidate) sentence and a reference (gold standard) sentence. METEOR uses:
- **Exact word matches**  
    (e.g., "dog" = "dog")
- **Stemming matches**  
    (e.g., "run" matches "running")
- **Synonym matches**  
    (e.g., "car" matches "automobile" using WordNet)
- **Paraphrase matches** _(in some versions)_  
    (e.g., "shut the door" matches "close the door")

METEOR align words from the candidate to the reference using the best matches from the 4 types above. Then it computes F-score weighted towards recall, then applies penalty for matching words that are out of order.

Pros: Better semantic matching than bleu
Cons: More computationally expensive, and requires resources like WordNet or/and paraphrase dictionaries.

# BERTScore
Measures **semantic similarity** between the generated text and a reference (ground-truth) answer using **BERT embeddings** — not just exact word overlap like BLEU or ROUGE.

- Pass both **generated** and **reference** texts through a pre-trained BERT model.
- Compute embeddings for each token.
- For each token in the candidate (generated) sentence, find the most similar token in the reference sentence (and vice versa).    
- Average the similarity scores.

Repeat: - For each **generated token embedding**, it looks for the **most similar reference token embedding** — **regardless of order**.  - Same in the other direction (reference to candidate).

Thus: any semantically similar token match contributes positively — order is **implicitly** handled only if BERT embeddings capture contextual differences. 

BERTScore works because it uses **BERT's contextualized embeddings**, which "know" that "play" can mean a verb or a noun depending on the surroundings.

There is no explicit order modeling though. 
# Metrics for RAG

**RAG** (Retrieval-Augmented Generation) is a class of LLM-based systems that **combines retrieval of external knowledge (from a document store or database) with text generation** to produce informed, contextually rich outputs.

Instead of generating purely from the model's parameters (like vanilla GPT models), RAG models fetch relevant documents at inference time and use them as additional context to guide generation.

In RAG (Retrieval-Augmented Generation): there are two "layers" of evaluation:
* Retrieval metrics: Recall@k, MRR -> These focus mostly on whether the correct documents were retrieved and fed to the generator. These metrics don't care about the final generated answer
* Generation metrics: Measure the quality of the final answer text produced from the retrieved documents: BLEU, ROUGE, BERTScore, Human Judgment.
## Recall@k
For each query, does at least one of the "ground-truth" relevant documents appear in the top **k** documents retrieved by the system? It is a **fraction (or %) of all queries where this happens.**

Example:

You have a **retrieval system** and 3 queries:

| Query ID | Ground-truth Relevant Docs | Retrieved Top-5 Docs           |
| -------- | -------------------------- | ------------------------------ |
| Q1       | Doc7                       | Doc2, Doc7, Doc9, Doc11, Doc15 |
| Q2       | Doc4                       | Doc1, Doc3, Doc5, Doc8, Doc9   |
| Q3       | Doc12                      | Doc12, Doc3, Doc4, Doc6, Doc9  |

Q1, Q3: Hit ; Q2: Miss ->  Recall@5 = 2/3 = 66.6%
"In 66.6% of the queries, the system retrieved the correct document within its top 5 retrieved results."

## MRR (Mean Reciprocal Rank)

Measures **how high the first relevant document appears in the ranked list of retrieved documents**.

Reciprocal Rank = 1 / (Rank of first relevant document)

# Factual consistency
Factual consistency in large language models (LLMs) refers to **the degree to which the information generated by the LLM aligns with known facts or information present in a given source document or context**. In simpler terms, it's about whether the LLM is telling the truth and not making things up that aren't supported by the evidence.

It's hard to capture though. Often involves a second LLM call judging whether there's been a hallucination.
# Non-functional metrics
- Inference latency
- throughput (tokens processed per second)
- CPU/GPU 
- Cost per query (USD)
# Loss functions

When we optimize neural networks (e.g. an LLM) we ultimately need to define a loss function and optimize it against some type of desired output. BLEU, ROUGE, etc are not suitable, they are non-differentiable to start with.

Instead, token-level cross-entropy loss is typically used. 

- You provide **input text** and the **expected output text**.
- The model's decoder generates one token at a time.
- For each position, the model outputs a **probability distribution over the vocabulary**.
- The cross-entropy loss compares the predicted distribution to the **ground truth token** at that position.
- The total loss is the **sum (or mean)** of cross-entropy over all tokens in the output sequence.

Why cross entropy?
Because almost all LLM tasks boil down (during training) to **next-token classification** or **label classification**:
- **Language modeling:**  
    Predict the **next token** given previous tokens (vocabulary classification problem).
- **Text Classification:**  
    Predict the **class label** (e.g., sentiment: positive/negative).
- **Machine Translation / Summarization / QA (Seq2Seq):**  
    Predict the **next token in the output sequence** given input and previous outputs.
All of these reduce to **probability distribution over the output space** (tokens or labels) — and Cross-Entropy measures how close this predicted distribution is to the "true" (one-hot) target distribution.