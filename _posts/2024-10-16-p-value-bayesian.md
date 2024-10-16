---
layout: post
title: "The Problems with p-values"
subtitle: "Why Frequentist Significance Testing Falls Short"
cover-img: /assets/img/posts/2024-10-16/coin-flip.webp
thumbnail-img: /assets/img/posts/2024-10-16/coin-flip.webp
tags: [statistics, p-values, data science, frequentist vs bayesian, statistical significance, hypothesis testing]
author: Taewoon Kim
mathjax: true
---

Statistical significance testing, specifically the use of p-values, has been the
cornerstone of hypothesis testing for decades. However, this frequentist approach has
critical flaws that can lead to misleading interpretations and false confidence in
research results. In this post, we will break down why p-values often fall short and
discuss how Bayesian methods can offer a clearer and more informative interpretation.

## Understanding p-Values: A Quick Recap

The p-value is defined as the probability of observing results at least as extreme as
the actual data, assuming the null hypothesis is true:

$$ p = P(\text{data} \mid H_0) $$

Where:
- $$ p $$ is the p-value,
- $$ H_0 $$ is the null hypothesis.

If the p-value is less than the predefined significance level (e.g., $$ \alpha = 0.05
$$), the result is labeled "statistically significant." However, this interpretation has
led to common misunderstandings.

## A Simple Example: The Binomial Test

Let’s take an example where we flip a coin that is very close to fair, with a true
probability of heads of 0.501. While this is practically a fair coin, let's see how
p-values behave as we flip it more and more times.

We flip the coin several times: $$ N = 10 $$, $$ N = 100 $$, $$ N = 1000 $$, $$ N =
10,000 $$, and finally $$ N = 1,000,000 $$. After each set of flips, we count the number
of heads and calculate the p-value to test whether the coin is significantly different
from a fair coin (i.e., $$ \theta = 0.5 $$).

Here are the results:

- For $$ N = 10 $$, we observed 6 heads and the p-value was **0.754**.
- For $$ N = 100 $$, we observed 51 heads and the p-value was **0.920**.
- For $$ N = 1000 $$, we observed 501 heads and the p-value was **0.975**.
- For $$ N = 10,000 $$, we observed 5010 heads and the p-value was **0.849**.
- For $$ N = 1,000,000 $$, we observed 501,000 heads and the p-value dropped to
  **0.046**.

### The p-Value Problem: It's Not What You Think

Notice what happened here: as we flipped the coin more times, the p-value gradually
decreased. By the time we flipped the coin 1,000,000 times, the p-value dropped below
the arbitrary threshold of $$ \alpha = 0.05 $$. According to the frequentist approach,
this result is "statistically significant," and one might conclude that the coin is not
fair. But really? Are we seriously going to argue that a coin with a 0.501 probability
of landing heads is significantly different from a fair coin in any meaningful sense?

This result highlights two fundamental problems:

1. **Arbitrary Threshold**: The $$ \alpha = 0.05 $$ threshold is completely arbitrary.
   We could easily make $$ p = 0.046 $$ insignificant by choosing a slightly stricter $$
   \alpha = 0.01 $$. There’s no solid reason why $$ 0.05 $$ should be the magic cutoff.
   
2. **p-Value Decreases with More Data**: As we increase the sample size, the p-value
   decreases—even though the coin remains practically fair. This means that for large
   datasets, trivial differences can become "statistically significant" even when they
   are meaningless in the real world.

## A Bayesian Approach: Learning from the Data

Now, let’s see how the Bayesian approach can provide a better understanding of the
coin’s fairness. Instead of calculating a p-value and making binary decisions, Bayesian
methods allow us to update our beliefs about the probability of heads as we observe more
data.

### Bayesian Updating

In Bayesian inference, we update our belief about the probability of heads, $$ \theta
$$, using observed data. This is done by combining a **prior distribution** (our initial
belief about $$ \theta $$) with the data to produce a **posterior distribution**—our
updated belief about $$ \theta $$ after seeing the data.

Mathematically, this is done using Bayes' theorem:

$$ P(\theta \mid \text{data}) = \frac{P(\text{data} \mid \theta)
P(\theta)}{P(\text{data})} $$

Where:
- $$ P(\theta \mid \text{data}) $$ is the posterior probability distribution for $$
  \theta $$, the probability of heads,
- $$ P(\text{data} \mid \theta) $$ is the likelihood of the observed data given $$
  \theta $$,
- $$ P(\theta) $$ is the prior probability distribution of $$ \theta $$,
- $$ P(\text{data}) $$ is the marginal likelihood of the data (a normalizing constant to
  ensure the posterior is a valid probability distribution).

In this context:
- $$ P(\theta) $$ represents our prior belief about the fairness of the coin before
  seeing any data,
- $$ P(\text{data} \mid \theta) $$ is the likelihood of observing the number of heads we
  got, given the coin’s true probability $$ \theta $$.

For a Bernoulli process like a coin flip, the posterior distribution for $$ \theta $$
follows a **Beta distribution**, which is the conjugate prior for the Bernoulli
likelihood. This means that after observing data, the posterior distribution remains in
the same family as the prior distribution, making it easy to update.

### Example: Bayesian Updating in Action

Let's say we start with a uniform prior $$ \text{Beta}(1, 1) $$. After observing each
set of flips, we update the parameters of the Beta distribution. For example, after $$ N
= 1000 $$ flips with 501 heads, the posterior distribution would reflect a high
confidence that the true probability $$ \theta $$ is close to 0.501. As $$ N $$
increases, the posterior distribution becomes increasingly concentrated around $$ \theta
= 0.501 $$.

In mathematical terms, the posterior distribution after observing $$ x $$ heads out of
$$ N $$ flips is:

$$ \text{Beta}(\alpha_{\text{posterior}}, \beta_{\text{posterior}}) $$

Where:
- $$ \alpha_{\text{posterior}} = \alpha_{\text{prior}} + \text{heads observed} $$
- $$ \beta_{\text{posterior}} = \beta_{\text{prior}} + \text{tails observed} $$

Here, $$ \alpha_{\text{prior}} $$ and $$ \beta_{\text{prior}} $$ come from the Beta
distribution used as the prior.

### Bayesian Advantage: Learning the True Distribution

Unlike the frequentist approach, where the p-value decreases as $$ N $$ increases, the
Bayesian approach naturally narrows down to the true probability. Instead of using
arbitrary thresholds, the Bayesian method allows us to learn more about the true
distribution of the coin's behavior as we collect more data.

As $$ N $$ increases, our posterior distribution becomes more sharply centered around $$
\theta = 0.501 $$, indicating that we are increasingly confident in the true fairness of
the coin. This is exactly what we expect: with more data, we should get a more accurate
estimate of the true probability.

Moreover, Bayesian methods allow for other advanced techniques like **Bayesian
Hypothesis Testing**, which can be used to directly compare hypotheses. But that is a
topic for a future post.

## Conclusion: Moving Beyond p-Values

The p-value-based frequentist approach to statistical significance is not only limited
but can also be misleading. In our coin flip example, increasing the number of flips
caused the p-value to drop below the arbitrary threshold of $$ \alpha = 0.05 $$, leading
to the incorrect conclusion that the coin is biased. In contrast, the Bayesian approach
allows us to continuously update our understanding of the coin’s fairness based on the
observed data, and as we gather more flips, we get closer to the true probability of
heads.

Bayesian methods offer a richer and more nuanced interpretation of data, free from the
arbitrary thresholds and misleading binary decisions that p-values impose. It's time to
move beyond p-values and embrace methods that allow us to learn from data in a more
flexible and informative way.
