---
tags: 
aliases: 
publish: false
slug: probability-recap
title: Recap on Probability for ML engineers
description: Recap on Probability for ML engineers
date: 2024-11-18
image: /thumbnails/backbones.png
---
# Intro

Probability is the mathematical study of uncertainty. Let's make a summary of basic concepts that should always be fresh in a machine learning engineer's mind.

# Definitions

* The **sample space** $\Omega$ is the set of possible outcomes of an **experiment**.
* Elements in $\Omega$ are called **sample outcomes** or **realizations**.
* Subsets of $\Omega$ are called **events**.

*E.g. toss a coin twice*
- $\Omega = \{ HH, TT, HT, TH\}$
- The first toss is head corresponds to the event $\{HT, HT\}$

The **Probability of an event A $P(A)$** is a real number. To qualify as a probability, P must satisfy three axioms:
1. $P(A) >= 0$ for every A
2. $P(\Omega) = 1$ 
3. If  $A, B$ are disjoint, then $P(A \cup B) = P(A) + P(B)$

In the discrete case, the probability distribution is given by a **probability mass function** p describing:

$P(X=x) = p(x)$
then it follows e.g. $P(\{x_1,x_2,x_3\}) = p(x_1)+p(x_2)+p(x_3)$

In the continuous case, the probability distribution is given by a **probability density function** p.
- The probability of X taxing a specific value is 0, because there are infinitely many possible values
- $P (a \leq X \leq b) = \int_a^b f(x) dx$
- the integral over -inf and +inf is 1

# Joint and conditional probabilities

The **joint probability** of two events X and Y is the probability of both happening at once $P(X,Y)$ or $P(X \cap Y)$.
- If X and Y are events on the same probability space, the intersection is actually set intersection
- If X and Y are actually events from different probability spaces, it's describing the relationship of two variables.

The **marginal probability** is the probability of a single variable in a joint distribution, e.g.:

$P(X=x) = \sum_{b \in \text{all values of }Y} P(X=x, Y=b)$

* For discrete variables thus, $P(x) = \sum_y p(x,y)$
* For continuous variables $p(x) = \int_y p(x,y) dy$

The **conditional probability** is the probability of one variable given that another variable takes a certain value that another variable takes a certain value

```
p ( X = x | Y = y ) = p ( X = x, Y = y ) / p ( Y = y )
```

Complement rule:  $P(\neg A) = 1 - P(A)$
Product rule:  $P(A, B) = P(A | B) \cdot P (B)$
Rule of total probability:  $P(A) = P(A, B) + P (A, \neg B)$

Generalizes to marginalization: $p(x) = \sum_y p(x,y)$
Which generalizes to **law of total probability**: $p(x) = \sum_y p(x|y) \cdot p(y)$

# Independence

A and B events are independent if:
* $P(A, B) = P(A) \cdot P(B)$
Equivalent to:
* $P(A | B) = P(A)$ and $P(B|A) = P(B)$

X and Y variables are independent if their joint distribution factorizes as a the product of their marginal distributions:
$P(X=x, Y=y) = P(X=x) \cdot P(Y=y)$ for all x, y

Two events A, B are **conditionally independent** given C if:

$P(A, B | C) = P(A|C) \cdot P(B|C) \Leftrightarrow P(A| B, C) = P(A|C)$

# Bayes rule

```
P(A|B) = P(A,B) / P(B)      (conditional probability)
P(A|B) = P(B|A) P(A) / P(B)   (product rule)
P(A|B) = P(B|A) P(A) / { sum_Ai  P(B|A_i)P(A_i)} (law of total probability)
```

Bayes' Rule allows us to update our beliefs about an event `A` (hypothesis) based on new evidence `B`. 

Imagine you're building a medical diagnostic system to determine if a patient has a disease (D) based on a test result (T).

- $P(D)$: The prior probability of having the disease (e.g., prevalence in the population).
- $P(T|D)$: The likelihood of the test being positive given the patient has the disease.
- $P(D|T)$: The posterior probability of having the disease given a positive test result.

Bayes' Rule allows you to correctly compute $P(D|T)$, which is crucial for informed medical decisions.

Remember, bayes rule lets you express:

$POSTERIOR\_PROBABILITY \propto LIKELIHOOD \times PRIOR\_PROBABILITY$

## Applications:

### Naive Bayes Classifier
Probabilistic models such as **naive bayes classifier**, which assumes conditional independence between features and the class label. It computes

$p(C | X_1, X_2, \ldots) = P(class) \cdot \prod_i P(X_i | C)$

Each $P(X_i | C)$ is estimated from training data, e.g. by fitting a gaussian or a frequentist approach for discrete variables.

### Graphical Models

**graphical models** where nodes represent random variables and edges represent conditional dependencies.  Each node has a conditional probability distribution representing the probability of a variable given its parents in the graph. E.g:

$P(A,B,C)=P(A) \cdot P(B|A) \cdot P(C|A,B)$

The edges in the graph specify which variables conditionally depend on others.

Markov Assumption: each variable is conditionally independent of all other variables in the network, except for its **parents** (direct predecessors in the graph). 

E.g. for graphslam, we have:

* Nodes represent the robot and landmark poses
* Edges represent odometry (pose to pose constriants) or observation (node to landmark constraints)
* The motion model is known $P(x_t | x_{t-1})$ typically a gaussian distribution
* The observation model is known:  $P(landmark | robot\_pose)$ and describes how the robot sensors perceive the landmarks

Graphslam tries to maximize the posterior  $P(X, L | Z)$ where Z are the sensor observations.

Bayes rule helps because we can derive $P(X | Z)$ from motion and sensor models.

### Hidden markov model

The system follows the **Markov property**, which means that the probability of being in state $S_t$ only depends on the previous state $S_{t-1}$ and not on earlier states.

**Hidden states** are unobserved (hence "hidden"), and we observe outputs (observations) that depend on those states.

The key task in HMMs is often to determine the most likely sequence of hidden states given the sequence of observations, which is a **posterior** distribution.

$P(S | O) \rightarrow P(O|S)$ emission probability $P(S)$ priors

Markov Chains

Markov chains are stochastic models used to describe a sequence of events where the probability of each event only depends on the previous event. They find applications in various fields, including natural language processing, where they are used for tasks like text generation and speech recognitio

# Probability distribution statistics

Mean:  $E[x] = \mu$
* Discrete variable:  $\sum_i x_i \cdot p(x_i)$
* continuous  $\int_{-\infty}^{+\infty} x \cdot f(x) dx$
Variance:   $E[(X-\mu)^2]$

# Popular distributions

## Discrete

Bernoulli: 
* Models a a binary variable with a parameter $\mu$
		$Bern(x|\mu) = \mu^x \cdot (1-\mu)^{(1-x)}$

Binomial distribution:
* We have N bernoulli trials
* $\mu$ is the probability of success of each trial
* m is the number of successes

![[Pasted image 20241218093510.png]]

Poisson distribution: Models the number of events occurring within a fixed interval of time or space, given that these events occur with a known constant rate and are independent of each other.

e.g.  you have 1 plane departing on average every 10 minutes ; it models $P(50  planes)$.
The **Poisson distribution** is **skewed** when $\lambda$ is small, because the probability of getting 0 or 1 event is high, but as $\lambda$ increases, the distribution starts to look more **symmetric** and approximates the Normal distribution (thanks to the **Central Limit Theorem**).

## Continuous

![[Pasted image 20241218094716.png]]
The **Normal distribution**, also known as the **Gaussian distribution**:

* The Normal distribution is **symmetric** around its mean.
* **68-95-99.7 Rule**:
	* About **68%** of the data in a Normal distribution falls within **1 standard deviation** from the mean.
	* About **95%** falls within **2 standard deviations** from the mean.
	* About **99.7%** falls within **3 standard deviations** from the mean.
* The Normal distribution has **thin tails**.
* In many real-world situations, data often appears to follow a Normal distribution, especially when the data is the result of a large number of independent, random factors combined.


One of the most important reasons for the prominence of the Normal distribution is the **Central Limit Theorem (CLT)**. The CLT states that, under certain conditions, the distribution of the sum (or average) of a large number of independent, identically distributed random variables will approximate a Normal distribution, regardless of the original distribution of the variables.

Suppose you're interested in understanding the average height of adult women in a country. 
- Let's assume that the distribution of heights of individual women is not Normal. It could be skewed, bimodal, or have some other complex shape.
- Now, you calculate the **mean height** of this sample of 30 women. The mean might still be a little skewed due to the small sample size and the underlying irregular distribution.
- - Next, you repeat this sampling process many times — say, 1,000 times. Each time, you randomly select 30 women and calculate the mean height for each sample.
- According to the **Central Limit Theorem**, the **distribution of these sample means** will start to look more and more **Normal** as you take more samples, regardless of the shape of the original distribution.

# Hypothesis testing

**Hypothesis testing** is a statistical method used to test assumptions (hypotheses) about a population using sample data.

## **Steps in Hypothesis Testing:**

1. **State the Hypotheses:**
    
    - **Null hypothesis (H₀):** This is the assumption you are trying to test. It's typically the "no effect" or "no difference" hypothesis.
    - **Alternative hypothesis (H₁ or Ha):** This is what you want to prove. It suggests that there is an effect, a difference, or some relationship.

E.g. $H_0: \mu=100$ $H_1: \mu \neq 100$

2. **Select the Significance Level (α):**

The **significance level (α)** is the probability of rejecting the null hypothesis when it is actually true (Type I error).
Common choices for $\alpha$ are 0.05 or 0.01, which means you are willing to accept a 5% or 1% chance of making a false positive error.

3. **Choose the Test Statistic:** The test statistic is a value that helps decide whether to reject the null hypothesis. It depends on the type of data and the distribution. e.g. z-test

![[Pasted image 20241218100129.png]]

4. **Calculate the p-value:** The **p-value** tells you the probability of observing the data you have (or something more extreme) assuming the null hypothesis is true. Usually you can lookup the p-value form the z value.

Then. $p-value \leq \alpha$ => reject null hypothesis  ; otherwise keep it.

## Confidence intervals

A **confidence interval (CI)** provides a range of values that is likely to contain the true population parameter (e.g., mean, proportion). It also gives us an idea about the uncertainty of our estimate.

1. **Choose the Confidence Level (e.g., 95%):**
-  A **95% confidence interval** means that if we were to take 100 samples, about 95 of those intervals would contain the true population parameter.
2. Calculate the Sample Statistic
3. Calculate the variability of the sample statistic e.g. for $\mu$

	SE =  mu / sqrt(n)

4. Calculate the confidence interval

![[Pasted image 20241218100450.png]]


![[Pasted image 20241218100542.png]]

## Types of error

- **Type I Error (False Positive):** Rejecting the null hypothesis when it is actually true.
    - Example: Concluding that a new drug is effective when it is not.
- **Type II Error (False Negative):** Failing to reject the null hypothesis when it is actually false.

## Hypothesis testing + Confidence intervals

- **Confidence intervals** give you a range of plausible values for a population parameter.
    
    The relationship between them is:
    
    - If a **null hypothesis** value falls **outside** the **confidence interval**, you **reject** the null hypothesis.
    - If the **null hypothesis** value is **inside** the **confidence interval**, you **fail to reject** the null hypothesis.

For the hypothesis test H0:$\mu=160$H the **95% confidence interval** for the population mean might be $[163.04, 166.96]$. Since 160 is **not within this interval**, you would reject H0​.

# Frequentist vs Bayesian

They are different interpretations of probabilities.

- **Frequentist** Probability is interpreted as the long-run frequency of events
- **Bayesian** Probability is interpreted as **degree of belief** or confidence in an event, and it can be updated as new evidence or data becomes available.


**The Law of Large Numbers** states that as the number of trials or observations increases, the sample mean approaches the population mean.

# Maximum Likelihood Estimation 


Maximum Likelihood Estimation is a method used to estimate the parameters of a statistical model. It seeks the parameter values that maximize the likelihood function, making the observed data the most probable. MLE is widely applied in machine learning for parameter estimation in models like linear regression and logistic regression.

The reason you don't always explicitly see probabilities or the likelihood function in standard machine learning problems (like linear regression or deep learning) is because the **loss functions** you're optimizing (e.g., Mean Squared Error for linear regression or cross-entropy for classification) are derived from **MLE**. In fact, the typical loss functions in machine learning are directly related to the log-likelihood of the underlying probabilistic model.

So, in linear regression, although you may not directly think in terms of probabilities, you are essentially performing **MLE** by minimizing the sum of squared errors. In other words, you are finding the parameter values (weights and bias) that maximize the likelihood of observing your data, given a Gaussian assumption about the noise.


# Practical exercises

## Example 1
- 60% of ML students pass the final and 45% of ML students pass both the final and the midterm 
- What percent of students who passed the final also passed the midterm?

```
P(Final) = 0.6
P(Final , Mid) =  0.45
P(Mid | Final) = P (Mid, Final) / P(Final) = 0.45 / 0.6  = 0.75
```

## Example 2

Marie is getting married tomorrow at an outdoor ceremony in the desert. In recent years, it has rained only 5 days each year. Unfortunately the weatherman is forecasting rain for tomorrow. When it actually rains, the weatherman has forecast rain 90% of the time. When it doesn't rain, he has forecast rain 10% of the time. What is the probability it will rain on the day of Maries' wedding?

P(rain=True) = 5/365
P(forecast=yes | rain=True) = 0.9
P(forecast=yes | rain=False) = 0.1

We are asked to calculate:

```
P(rain = True | forecast = yes)  = 
P(forecast = yes | rain = True) * P(rain = yes) / P(forecast = true) = 
( 0.9 * 5/365 ) / { 0.9 * 5/365 + 0.1 * 360/365 } =   0.111
```

So, despite the gloomy prediction, it is unlikely it's going to rain!

# Glossary

- Coefficient of variation (CV): Metric of dispersion.
Sdev / mean => variability in unit of means.

* Coefficient of determination `R^2` is  a performance measure for regression.

`R^2 = 1- SS_residual/SS_total`

where

- SS_residual: sum of squared residual errors bw observed and predicted values  (the difference between observed and predicted values).
- SS_total: (the difference between observed values and the mean of observed values).

represents the proportion of the variance in yyy that is explained by the independent variables in the model. 

* ROC ciurve ir TPR = Recall = Sensitivity= TP/(TP+FN) and FPR =  FP / FP+TN
* AUC: Area under it
* Alternative you use P-R curv

Cross entropy:

![[Pasted image 20241218103938.png]]

-y * log(p) - (1-y) * log(1-p)

may not be severe enough if it is too imbalanced
you could use weights.

![[Pasted image 20241218104000.png]]

Soft voting: average of probabilities
Hard voting: majority

lasso (L1)  vs ridge (L2) so basically you add a regularization term to your loss where you try to shrink the weights. if you use l1-norm or l2-norm differentiates these two approaches
l1 takes too many of them to 0

elastic = l1 and l2 regularization

bootstrap: resampling technique
boosting: train a model on th edata. for those samples misclassified, add another model that focuses more on that misclasified part. - Boosting gives more weight to samples that are misclassified in each round. As a result, the algorithm focuses on the hard-to-classify examples, trying to improve the model iteratively.

bagging:  multi many trees, trained on random subsets of the dataset. average all predictions or vote at the end.
