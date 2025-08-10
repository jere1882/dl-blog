---
tags:
  - deep-learning-basics
aliases: 
publish: true
slug: a-comprehensive-introduction-to-transformers
title: A comprehensive introduction to transformers
description: Notes and a comprehensive summary of basic concepts in the field of transformers
date: 2024-07-07
image: /thumbnails/pick_architecure.jpeg
---
This post is an organized compendium of personal notes, excerpts from other posts and papers, which I’ve assembled into my own in-depth introductory guide to transformers. My goal is to understand the basic building blocks of these architectures and how they are used in cutting-edge modern systems.

The reason I decided to revisit my foundations in transformers is that I attended ICRA 2023 (arguably the most important robotics conference in the world!), and literally half the papers were about using transformers for virtually every imaginable robotics task. It made me realize that transformer architectures may well come to dominate the field in the decades to come.

Let’s dive in!
# Introduction

Transformer architectures have been in the spotlight for the last six years, ever since the groundbreaking paper "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" introduced the Transformer model in 2017. Even though they were initially designed for natural language processes, transformers have made significant inroads into computer vision.

A transformer is an attention-based encoder-decoder neural network architecture that was developed to deal with sequential data, as RNNs and LSTMs do.
![Pasted image 20240729043308](/assets/Pasted%20image%2020240729043308.png)
*Image from the original paper*

The encoder (left side) maps an input sequence into an abstract continuous representation that holds all the learned information of the input sequence.

The decoder takes the continuous representation and, step by step, generates a single output, while also being fed its own previous output.

# The original transformer architecture in depth

Let's zoom into the details. This section is adapted from this great [post]([https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0](https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0)).

## The encoder and the attention mechanism

(1) The input layer of the encoder converts each word in the input sequence into a fixed-length **embedding** representation. This layer is learned.

![Pasted image 20240729043633](/assets/Pasted%20image%2020240729043633.png)

(2) Positional information is injected into the embeddings. This is necessary because the transformer encoder doesn't have recurrence like RNNs, so we need to somehow specify the position of each word in the input.

This is done by a trick called **positional encoding**, which consists of adding stuff into each embedding and the position ends up encoded there.

(3) The encoder layer itself. Its function is to map input sequences into abstract output sequences that hold the learned information:

![Pasted image 20240729043918](/assets/Pasted%20image%2020240729043918.png)
> 💡 **Remark:** In this example, you have an input sequence of size 4 and embedding size 3. Notice that the input of the encoder is thus a tensor of size 4x3, and the output of the encoder is a tensor of the exactly same size. The Multi-Headed Attention also returns a tensor of the same shape as the input.

The **Multi Headed Attention** block applies a specific attention mechanism called **SELF-ATTENTION**. Self attention allows the models to associate each word in the input to other words in the input.
* We use residual connections
* Feed forward is just a fully connected NN with ReLu activations

(3.1) The attention layer

To achieve self-attention, we take each embedding in the input sequence and map it (via Linear layers) to a **query**, **key** and **value** vectors.

![Pasted image 20240729044229](/assets/Pasted%20image%2020240729044229.png)

All Q K V then go through the following module:

![Pasted image 20240729044300](/assets/Pasted%20image%2020240729044300.png)

They get fed to a linear layer,  Q and K undergo a dot product to produce a **score** matrix. This matrix basically determines how much focus should a word put into another words.

![Pasted image 20240729044347](/assets/Pasted%20image%2020240729044347.png)

The scores are scaled down to stabilize the gradients, and softmax is applied to map the scores to the range 0-1. The result are the **attention weights**, which tell you for each pair of words in the input sequence, their relative attention.

Finally, the attention weights are multiplied by the values to get the attention block output.

![Pasted image 20240729044548](/assets/Pasted%20image%2020240729044548.png)

> 💡 **Analogy**: Imagine you're in a library. You want to find a specific book (your query). You know the book's title and and author (your key). You search through the library catalog (a collection of keys and values). The value is the actual book.

**Key point: The semantic meaning of the output of the attention mechanism**

*Attention* doesn't change the shape of the data, but it drastically changes its _content_. The new content is:
- Each element in the output is a weighted sum of all input elements.
- The weights are determined by how relevant each input element is to the current position.

**Example:**

- In a machine translation task, when translating the word "king," the attention mechanism might focus on the words "royal" and "man" in the input sentence, giving them higher weights. This helps the model capture the meaning of "king" more accurately.
- In a text summarization task, the attention mechanism can focus on the most important sentences to generate a concise summary.

**This is achieved by:**

The Q, K, V mechanism might seem arbitrary at first glance. However, it's a clever way to introduce flexibility and learning capacity into the attention process.

- **Query (Q):** Essentially, the query represents what you're looking for in the sequence. It's a vector that defines the search criteria. By learning different query representations, the model can focus on different aspects of the input.
- **Key (K):** Think of the key as a representation of the information contained in each element of the sequence. It's what the model uses to determine how relevant each element is to the query.
- **Value (V):** The value is the actual information associated with each element. It's what the model will use to construct the output based on the attention weights.

The math works as follows:

Remember, the target output is a weighted sum of the values, where the weights are determined by the similarity between the query and keys.

1. The dot product between Q and K measures their similarity. A higher dot product indicates greater similarity. This is a common technique used in ML to measure the similarity between vectors.  

 *If you have embedding size 32 and sequence length 10, Q and K have shape (10,32).  We calculate `Q x K.T`, obtaining a (10,10) matrix. Element i,j  gives you the "similarity" between the ith and jth elements of the sequence. *

![Pasted image 20240731010748](/assets/Pasted%20image%2020240731010748.png)
2. Softmax normalization:  Converts raw similarity scorss into probabilities, ensuring that the weights sum up to 1 on each row.
3. Multiplication of attention weights by V. This computes a weighted sum of the value vectors, where each weight is determined by the attention scores.

```
     (   ) 32
      10
10 (  )    (  ) 32
   10       10

remember that matrix multiplication can be seen as a bunch of matrix-vector multiplication, one for each column.
```


(3.2) Multi head attention layer

We usually use `N` heads, which means that we calculate for each embedding in the input sequence `N` Queries, Keys and Values. Each Q K and V tuple goes through self-attention process individually, and the output of all heads gets concatenated before going through a final linear layer. In theory, each head would learn something different.

(3.3) The rest of the encoder

The rest of the layers and connections used (residual connections to help the network train ; fully connected layers with ReLus and normalization ) are standard deep learning elements.

The final output of the encoder is therefore a continuous representation with attention information. This will help the decoder focus on the appropriate words in the input during the training process. Furthermore, you can stack `M` encoder layers to further encode information, where each layer has the chance to learn potentially different attention representations.

## The Decoder

The job of the decoder is to generate text sequences, one token at a time. It encompasses similar layers as the encoder, and a linear layer that acts as a classifier on the last layer, using a softmax to get word probabilities.

*ASK: So, the output are not tokens, are words? Or letters? That'd mean a HUGE lot of output neurons...*

The decoder is **autoregresive**. It takes the encoder output as its intput, as well as all the previous decoder outptus generated so far (or just a start token if it's the first one). Then, it generates the next token (word).

![Pasted image 20240807141623](/assets/Pasted%20image%2020240807141623.png)

E.g:
1. Feed the encoder output and `start` token. Decoder outputs "I"
2. Feed encoder output and `start` "I". Decoder outputs "am"
3. Feed encoder output and `start` "I" "am". Decoder outputs "fine"
4. Feed e.o. and `start` I am fine. Decoder outputs `end` token.

*ASK: The encoding takes place just once right? And hten the auto-regression takes place with the encoding part fixed.*

Now that we understand what are the encoder inputs and outputs, let's look at the decoder structure itself.

![Pasted image 20240807142031](/assets/Pasted%20image%2020240807142031.png)

1. Just like the encoder, the decoder embeds the inputs (sequence generated so far) and applies positional encoding. 

*This is a learned embedding and it is not the same as the one learned by the encoder. The source (input) and target (output) sequences may come from different vocabularies, especially in tasks like machine translation. Even if the source and target languages are the same, they are typically treated as separate vocabularies for flexibility and efficiency.*

2. We pass these embedding into multi head attention layer. The only difference with the multi headed attention of the encoder is that we may apply **masking**, meaning that we do not let elements later in the sequence access elements older than in the sequence.

*The masking in the multi-headed attention layer of the decoder is crucial for ensuring that the decoder generates the output sequence correctly, particularly during training. This process is known as "masked multi-headed attention". In the decoder, masking is used to prevent the model from "seeing" future tokens in the target sequence. This is important because, during training, the model should only use the known information up to the current position to predict the next token. Masking ensures that each position in the output sequence can only attend to previous positions and itself, but not to any future positions.

Example:

- **Input Sequence (Source Sentence)**: "hi, how are you?"
- **Target Sequence (Expected Output Sentence)**: "good, and you?"

During training, we provide the target sequence to the decoder, but we want the model to learn to generate it one token at a time, without looking ahead at future tokens. This is where masking comes in.
When computing the attention for each position, the model can only attend to the previous tokens and itself. This is enforced using a mask.

In the second step of generation:
- **Input to Decoder**: "`<start>` good"
- **Mask Applied**: The "`<start>`" and "good" tokens are visible.
- **Attention Computation**: The model computes attention scores for "good" based only on "`<start>`" and "good".
- Attention for "good"=`softmax([0,0,−∞,−∞,−∞])`

**Without Masking**: If there were no masking, the model could potentially use information from future tokens to predict the current token, which would not be representative of the true sequence generation process.

How is it implemented?? Add this matrix M to the attention scores matrix

![Pasted image 20240807182958](/assets/Pasted%20image%2020240807182958.png)
3. The second multi head attention layer. Here, the encoder’s outputs are the **keys** and the **values** ; while the first multi headed attention layer outputs are the **queries** .

*Intuition: The encoder output serves as the "memory" that the decoder will refer to at each step of generating the output sequence. The K are used to match the Q from the decoder to find relevant information, and the V contain the actual contextual information that will be used by the decoder to generate the next token.*

*By using the encoder output as both keys and values, the decoder can access any part of the input sequence's contextual representation to gather the necessary information for generating the current output token.*

*The queries in the decoder come from the previously generated tokens (or the provided target tokens during training). These queries represent the current state or focus of the decoder — essentially what the decoder is trying to generate next, based on what it has generated so far. The queries are used to search the encoder's output (keys) to find the most relevant parts of the input sequence that can help generate the next token in the output sequence.*

*Note: The encoder is run just once to get the encoded input sequence. Then, the decoder will consume the same encoded sequence over and over to create each token one at a time. Encoding happens just once.*

5.  The output of the second multi-headed attention goes through a pointwise feedforward layer for further processing .

*in the context of the Transformer architecture, a "pointwise feedforward network" is essentially the same as a fully connected (or dense) neural network applied independently to each position (or token) in the sequence. It consists of two linear layers with a ReLU activation in between.*

6. Then a linear layer that acts as a classifier. The classifier is as big as the number of classes you have. E.g. 10K classes for 10K words. You apply a softmax to produce probability scores, and pick the most probable word as the final output.

*In this context of transformers for NLP, tokens and classes refer to words, or subwords, or characters.* 

# Variations

I have just described the transformer architecture as introduced in the original paper. 

There are many variations to the transformer block. The order of the various components is not set in stone; the important thing is to combine self-attention with a local feedforward, and to add normalization and residual connections.

![Pasted image 20240807185246](/assets/Pasted%20image%2020240807185246.png)

A transformer architecture is defined roughly as an architecture to process  a connected set of units (being tokens in a sequence, pixels on an image, etc) such that the only interaction between units is through self-attention.

# Pretraining

Pretraining is a crucial phase for modern architectures like BERT and GPT. It involves training a model on a large corpus of text to learn general language representations before fine-tuning it on specific tasks. The pretraining methods leverage large amounts of text data and typically involve tasks designed to help the model understand language structure and context. 

Sample tasks are:
* Masked language modeling: Predict missing words in a sequence
* Casual (Auto-regressive) language modeling: Predict next word in a sequence.
* Next sentence prediction: To predict if one sentence follows another in a coherent text.

Data used for pretraining could be:
* Large corpus from books (BookCorpus)
* Wikipedia
* news sites
* Tweeks

**Purpose**: To learn comprehensive language representations, which can be fine-tuned for specific downstream tasks.

![Pasted image 20240807190541](/assets/Pasted%20image%2020240807190541.png)
# A glimpse at modern transformer architectures 

Many famous architectures can now be briefly described.

## Bert (Bidirectional Encoder Representations from Transformers)

From 2018. A simple stock of transformer blocks (as described above), pretrained on a large general-domain corpus consisting of 800M words from English Books, and 2.5B words from English Wikipedia. It introduced a few original pretraining tasks.

Unlike the original Transformer encoder which processes text in a unidirectional (left-to-right) manner, BERT uses bidirectional context, allowing it to understand words based on their surrounding words in both directions.

*The original transformer we described above presents the self attention mechanism that is already able to attend to words before and after a given word in the input sequence. BERT is specifically designed to leverage this in a more structured way by using specific pretraining tasks that force the model to rely on what's after and before e.g. predict the missing word. So, the original Transformer encoder does indeed use bidirectional attention in practice, but BERT’s pretraining methodology with MLM explicitly enhances its ability to leverage and understand bidirectional context.*￼￼

* Human level performance on a variety of language tasks such as question answering, sentiment classification
* Uses WordPiece tokenization, which is somewhere in between mord-level and character-level sequences.
* The largest BERT model uses 24 transformer blocks, an embedding dimension of 1024 and 16 attention heads, resulting in 340M parameters.

## GPT (Generative Pretrained Transformer)
From 2018. GPT's approach is notable for its exclusive focus on generative tasks, extensive pretraining, and the use of a large-scale language model to achieve versatility and high performance in text generation and other NLP tasks.

GPT is based solely on the decoder component of the Transformer architecture. It uses a stack of transformer decoders without an encoder. GPT employs a causal (auto-regressive) language model that predicts the next word based on the preceding words in the sequence. This is similar to the unidirectional attention mechanism used in the original Transformer decoder but applied as the sole component of the architecture.

GPT is pretrained on a large corpus using a language modeling objective where it learns to predict the next token in a sequence. This pretraining involves learning from a vast amount of text data to develop general language understanding and generation capabilities.

*if it doesn't have an encoder, how does it process and integrate information from whatever input sequence or prompt it receives?*


GPT, despite not having a separate encoder component like the original Transformer architecture, processes and integrates information from input sequences or prompts through its stack of transformer decoders:

*I need to see a picture of the architecture to better understand it. It's unclear, are prompts and the sequence generated so far encoded and integrated? when and how*

## Subsequent versions
| Architecture        | Year | Key Contributions                                                                                                |
| ------------------- | ---- | ---------------------------------------------------------------------------------------------------------------- |
| **GPT‑2**           | 2019 | First widely-known autoregressive model with fluent long-form text generation; sparked interest in scaling laws. |
| **T5**              | 2019 | Introduced the “text-to-text” paradigm; influential in unifying NLP tasks under a single framework.              |
| **GPT‑3**           | 2020 | Breakthrough in scale (175B), strong zero-shot and few-shot performance, brought LLMs to mainstream awareness.   |
| **PaLM**            | 2022 | Pushed scaling further (540B), demonstrated multilingual and logical reasoning gains.                            |
| **ChatGPT**         | 2022 | First conversational LLM interface based on GPT‑3.5; made LLMs accessible to the public.                         |
| **GPT‑4**           | 2023 | Multimodal (text + vision), large context window, major leap in reasoning and reliability.                       |
| **Claude 2**        | 2023 | Strong alignment and safety emphasis; competitor to GPT‑4 in quality.                                            |
| **Gemini 1.5**      | 2024 | Introduced million-token context windows, high-quality multimodal capabilities from Google DeepMind.             |
| **GPT‑4o**          | 2024 | “Omnimodel” with real-time text, vision, and audio; very fast and cost-efficient.                                |
| **Claude 3.5**      | 2024 | Advanced reasoning and transparency tools; top-tier coding and math performance.                                 |
| **Gemini 2.5**      | 2025 | Google’s “thinking” model with internal reasoning steps; strong in multimodal tasks and robotics.                |
| **GPT‑4.5 (Orion)** | 2025 | Latest OpenAI flagship; larger model, excellent reasoning and world knowledge. Currently in research preview.    |

As of **mid‑2025**, the LLM landscape has **consolidated** around **three dominant model families**, each led by a major player:

|Family|Organization|Key Strengths|Flagship Models|
|---|---|---|---|
|**GPT**|OpenAI|Scale, general performance, multimodal speed, ecosystem|GPT‑3 → GPT‑4 → GPT‑4o → GPT‑4.5|
|**Claude**|Anthropic|Safety, transparency, reasoning depth|Claude 1 → 2 → 3 → 3.5 → 3.7|
|**Gemini**|Google DeepMind|Multimodal context, tool integration, robotic/agent use|Gemini 1 → 1.5 → 2.5|
This post goes on in my LLM architectures post :) !