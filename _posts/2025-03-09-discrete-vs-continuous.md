---
layout: post
title: "Discrete vs. Continuous: A Tale of Two Approaches to Language Modeling"
subtitle: "Balancing n-gram models and neural networks for next-token prediction"
cover-img: /assets/img/posts/2025-03-09/search.webp
thumbnail-img: /assets/img/posts/2025-03-09/search.webp
tags: [language-modeling, discrete-search, neural-networks, chain-of-thought, n-gram]
author: Taewoon Kim
mathjax: true
---

Modern Natural Language Processing (NLP) revolves around **language modeling**—the art
of predicting the next token given the previous ones. Formally, if we have a sequence of
tokens $$w_1, w_2, \dots, w_{n-1}$$, we want to learn:

$$ P(w_n \mid w_1, w_2, \ldots, w_{n-1}). $$

This post explores **two broad ways** to tackle this problem:

1. **Discrete Search (e.g., n-gram models)**
2. **Continuous Search (e.g., neural networks)**

We’ll see why discrete methods can be both intuitive and limited, and how continuous
approaches help overcome these limits—yet still rely on discrete search **after**
training for certain inference routines.

---

## 1. What Is a Language?

In **formal language theory**, a language $$L$$ is defined as a **set of valid strings**
over an **alphabet** $$\Sigma$$. For instance, if $$\Sigma = \{\texttt{0},
\texttt{1}\}$$, then a language $$L$$ might be all binary strings of even length, or all
binary strings containing an equal number of 0’s and 1’s, etc. Formally:

$$ L \subseteq \Sigma^*, $$

where $$\Sigma^*$$ denotes all possible finite sequences of symbols from $$\Sigma$$.
This **discrete** view underlies much of computational linguistics: we think of words
(or tokens) as **symbols** that combine into valid (or invalid) expressions.

---

## 2. Discrete Search via N-grams

### 2.1 N-gram Language Modeling

One classic way to capture a language model in a **purely discrete** form is the
**n-gram** approach. Suppose we have a sequence of words $$w_1, w_2, \dots, w_t$$. We
assume each word $$w_t$$ depends only on the previous $$(n-1)$$ words:

$$ P(w_t \mid w_1, \ldots, w_{t-1}) \;\approx\; P\bigl(w_t \,\mid\, w_{t-1}, \,w_{t-2},
\,\dots,\, w_{t-(n-1)}\bigr). $$

In practice, we **count** how often each sequence of $$n$$ words appears in our training
corpus. The **maximum likelihood estimate (MLE)** for the n-gram model is:

$$ P_{\text{MLE}}\bigl(w_t \mid w_{t-1}, \ldots, w_{t-(n-1)}\bigr) \;=\;
\frac{\text{Count}\bigl(w_{t-(n-1)}, \ldots, w_{t-1}, w_t\bigr)}
{\text{Count}\bigl(w_{t-(n-1)}, \ldots, w_{t-1}\bigr)}. $$

Why is this MLE? Because under a simplifying assumption that each observed n-gram in the
training corpus is generated from the same underlying distribution, counting relative
frequencies directly maximizes the likelihood of the observed n-gram counts.

### 2.2 Limitations of N-gram Models

#### **Combinatorial Explosion**  
N-gram models suffer from an exponential growth in the number of possible sequences as
the vocabulary size ($$V$$) and n-gram order ($$n$$) increase. Given $$V$$ distinct
tokens, the number of possible n-grams is $$ V^n $$, which quickly becomes infeasible
for even moderate values of $$n$$. For example, if $$V = 50,000$$ (a typical vocabulary
size for large-scale models):

- **Bigrams ($$n=2$$)** → $$50,000^2 = 2.5 \times 10^9$$ possible bigrams.
- **Trigrams ($$n=3$$)** → $$50,000^3 = 1.25 \times 10^{14}$$ possible trigrams.
- **4-grams ($$n=4$$)** → $$50,000^4 = 6.25 \times 10^{18}$$, an astronomically large
  number.

Storing and computing probabilities for every possible n-gram requires enormous memory
and computational resources, making this approach infeasible for large vocabularies and
longer context windows.

#### **Context Limitations**  
Natural language is highly **context-dependent**, but n-gram models are limited to a
**fixed-length** history of $$n-1$$ tokens. This makes them incapable of capturing
long-range dependencies. Consider the following sentences:

- *"The concert I went to last night was amazing!"*
- *"The show I attended yesterday was fantastic!"*

Both convey the same meaning, but an n-gram model would treat them as completely
different because they use different words outside the fixed n-gram window. As a result,
n-gram models struggle with **paraphrasing, synonymy, and flexible word order**, all of
which are common in natural language.

In contrast, modern neural models (e.g., Transformers) can track meaning over
arbitrarily long contexts using self-attention and continuous vector representations.

#### **Data Sparsity**  
Even with large corpora, the number of possible n-grams is so vast that most valid
sequences **will never appear** in the training data. This creates **zero probability**
problems—if a particular n-gram is missing from the training set, the model assigns it a
probability of zero, even if it is a perfectly reasonable phrase.

#### **Why Storing All Possible Sentences is Stupid**  
A fundamental weakness of n-gram models is that they store **only exact word sequences**
rather than understanding **semantic similarity**. Natural language allows for infinite
rephrasings of the same idea:

- **Synonyms**: *"happy"* vs. *"joyful"*, *"big"* vs. *"large"*.
- **Flexible word order**: *"I love programming"* vs. *"Programming is something I
  love"*.
- **Contextual variation**: *"John said he would come"* vs. *"John mentioned that he
  would arrive"*.

An n-gram model treats all these variations as **completely separate sequences**,
failing to recognize that they express the same meaning. This makes discrete sequence
storage **wasteful** and **ineffective** for real-world language modeling.

In contrast, modern deep learning models **embed words in continuous vector spaces**,
allowing them to generalize beyond exact matches and understand meaning at a more
abstract level. This fundamental shift—from **storing exact sequences** to **learning
patterns in continuous space**—is why neural networks have largely replaced n-gram
models in NLP.

In short, while **discrete search** in n-gram models is a simple and interpretable
approach, it struggles with scale, context, and generalization—making it inadequate for
modeling the complexities of human language.

---

## 3. Enter the Continuous Search: Neural Networks

### 3.1 Learning Parameters via Gradient Descent

Modern **deep language models** (e.g., Transformers) embed each word into a **continuous
vector space** and learn a **parametric function** $$f_\theta$$ that predicts the
probability of the next token:

$$ P_\theta(w_n \mid w_1, \ldots, w_{n-1}) \;=\;
\frac{\exp\!\Bigl(\mathbf{z}_\theta(w_1,\ldots,w_{n-1}, w_n)\Bigr)} {\sum_{w'}
\exp\!\Bigl(\mathbf{z}_\theta(w_1,\ldots,w_{n-1}, w')\Bigr)}, $$

where $$\mathbf{z}_\theta$$ is a learnable scoring function (often a neural network). We
fit the model’s parameters $$\theta$$ by minimizing the **cross-entropy loss** on the
training data. In expectation form:

$$ \mathcal{L}(\theta) \;=\; \mathbb{E}_{(w_1,\dots,w_n)\,\sim\,\text{data}} \bigl[-\log
P_\theta(w_n \mid w_1,\dots,w_{n-1})\bigr]. $$

Minimizing this loss with **gradient descent** is effectively a **continuous search**
over the high-dimensional parameter space to find $$\theta$$ that best fits the observed
text. Note that $$f_\theta$$ can be any function approximator. We used to use RNNs,
e.g., LSTMs, but now Transformers have become the most powerful neural network
architecture to approximate and train this function. This might of course change in the
future. 

### 3.2 Why Continuous Helps

1. **Generalization**: Similar words get similar embeddings, helping the model better
   predict unseen combinations.
2. **Handling Longer Context**: Transformer-based models can attend to hundreds or
   thousands of tokens, capturing rich dependencies beyond a fixed-window n-gram.
3. **Scalability**: Deep nets handle massive corpora better than naive n-gram counts.

---

## 4. Discrete Search Still Matters

Even though **continuous search** (e.g., gradient-based methods for neural networks) has
become the gold standard for training large-scale language models, **discrete search**
remains crucial during inference and structured reasoning tasks. 

For example, when generating text, models do not simply sample the most probable token
at each step; instead, they often perform **beam search**, a discrete search algorithm
that keeps track of multiple high-scoring sequences and expands them in parallel. This
allows the model to generate more **coherent and globally optimal** outputs instead of
being greedily biased toward the most likely next token.

Similarly, in reasoning-based tasks, such as **chain-of-thought**, **tree-of-thought**,
or **graph-based reasoning**, we can represent partial reasoning steps as **discrete
nodes**, with edges denoting possible transitions between them. The goal is to **search
for the best path** through this space to generate a logically sound conclusion.
Although the model assigns transition probabilities using continuous parameters, the
**search itself remains discrete**.

Thus, despite the dominance of continuous optimization in model training, discrete
search still plays an essential role in **inference, structured reasoning, and
decision-making**.

---

## 5. Conclusion

- **Language** can be viewed as a **discrete system** of symbols and strings.
- **N-gram models** represent a **purely discrete** approach via counts, which faces
  combinatorial blow-up and context limitations.
- **Neural networks** perform a **continuous search** over parameters (via gradient
  descent) to better handle large vocabularies and long context windows.
- **Discrete search** reappears at inference time, where techniques like **beam search**
  help optimize generated sequences, and structured reasoning benefits from **discrete
  path exploration**.

In modern computer science, **both discrete and continuous search** co-exist, and which
approach dominates depends on the problem at hand. Sometimes, the best solution involves
a hybrid approach—using neural networks to learn **continuous representations** and
discrete algorithms for **explicit reasoning and structured decision-making**.

Feel free to leave questions or thoughts in the comments section—happy language
modeling!
