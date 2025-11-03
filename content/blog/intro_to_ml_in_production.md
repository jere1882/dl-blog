---
tags:
  - Deep
  - Learning
  - Basics
aliases: 
publish: true
slug: intro_to_ml_in_production
title: Review and Summary of the course "Introduction to Machine Learning in Production"
description: Notes, assignments and review of the course
date: 2024-07-09
image: /thumbnails/coursera_dl_specialization.png
---
Here I'll post summaries from each section.

# Week 1 - ML Project Lifecycle ; Deployment and Monitoring

## The ML Project Lifecycle

There is so much more around building machine learning systems in production than writing code to train and evaluate models:
![[Pasted image 20241018034122.png]]

The **ML project lifecycle**  shows us the steps involved in bringing a ML model to production:

![[Pasted image 20241018034219.png]]

1. ***Scoping**: Decide what do you want to apply ML to. What’s X and what’s Y.*
2. *Collecting the **data**.*
3. ***Train** the model. This is **iterative**, often going back to 2.*
4. ***Deploy** and **monitor** in production. You may collect even more data from prod or you may need to adjust stuff, **going back** to previous items.*

*Example: Speech recognition*

*Scoping*
- *Decide to work on speech recognition for voice search*
- *Decide on key metrics: accuracy, latency, throughput (queries per second that can be handled)*
- *Estimate resources and timeline*
*Data*
- *Ensure consistent labeling*
- *Define volume normalization, silence before/after clip fix*
- *Ensure high quality data by using the proper frameworks*
- *Collect the data*
*Modelling:*
- *Code (algorithm/model/architecture)*
- *Tune hyperparamaters*
- *Often use an open source implementation*
- *Find out where model falls short, often use that to improve your data*
*Deployment:*
* *A structure like the following may be used*
* Monitor and find issues, e.g. young people using the model more but the model trained in adults. Go back and fix it (concept/data drift).
![[Pasted image 20241018034423.png]]

***MLOps* (Machine Learning Operations)** is an emerging discipline, and comprises a set of tools and principles to support progress through the ML project lifecycle.

The key idea in MLOps is the systematic ways to think about scoping, data, modeling, and deployment, and also software tools to support the best practices.The key idea in MLOps is the systematic ways to think about scoping, data, modeling, and deployment, and also software tools to support the best practices.

## Deployment

In this course, we'll study the lifecycle steps starting from the end: deployment and monitoring.

**_Deployment_** refers to the process of integrating a trained machine learning model into a production environment where it can make predictions or automate tasks in real-world scenarios. This step allows the model to be used by end-users, applications, or systems.

### Challenges in deployment

**Concept drift / data drift:** e.g. lighting of photographs changing in the wild different to what you trained with or a different microphone in a new phone. The data changes in some way.
- Gradual change -> English language evolves slowly
- Sudden shock -> COVID impact in sales
- Concept drift: the mapping x->y changes )e.g. house prices changing) vs Data drift: x changes (e.g. images now have a different resolution)
- One must be able to detect and correct any changes.

**Software engineering issues / decisions:**
- Do you need realtime predictions? Or are batch predictions OK? (like collect queries and run overnight batch records at night in a hospital)
- Does the prediction service run in the cloud ? Or does it run on an edge device? Maybe even in a web browser?
	- Cloud: more compute resources but requires internet connection
- How much compute resources you have? CPU/GPU/memory
- Latency and throughput (Queries per second QPS). E.g. requirement 500 ms to get a response to the user, 1000 QPS.
- Logging: For analysis and maybe collect more data for retraining
- Security and privacy
### Deployment techniques

Common deployment scenarios:
* Add a new product or capability
* Automate or assist on a task that so far has been being done manually
* Replace a previous ML system by a new one

Key ideas:
* Gradual ramp up of traffic with monitoring
* Rollback utilities so that you can go back to previous versions if needed

**Deployment patterns:**

 **Shadow Mode**
 - ML system runs in parallel to the human for a while (in his/her shadow!)MLOps, and its output is not used for any decisions.
- Allows for gathering data of how the algorithm is performing and how that compares to human judgement.
- Great way to verify the performance of a learning algorithm before letting it make any real decisions.
**Canary Deployment**
* Do this once you decided you will use the ml model.
* Roll out to a small fraction (say 5%) of traffic initially
- Monitor system and ramp up traffic gradually
**Blue-Green development**
* You have a router pointing at a blue model, and may change the model just by making the router point to the green model (prediction service)
![[Pasted image 20241018040630.png]]
* You can easily roll back by pointing back to the blue
* You can use gradual version slowly sending traffic to the green one

The idea is to move from left to right in terms of degree of automation
![[Pasted image 20241018171600.png]]Human in the loop: AI assistance and partial automation (just ask human if model is undecided)

## Monitoring

### What to monitor

Use a dashboard to track how it’s doing over time, tracking values indicative of things that can go wrong such as:

![[Pasted image 20241018171633.png]]

Tip: Set thresholds for alarms!

### Why

Models may need to be retrained (manually or automatically) and redeployed, software engeneering pieces may need to be improved, etc. Only by monitoring the system can you realise you need to do this.

### Monitoring pipeline systems

Many ML systems are not just a single model but a pipeline of steps. How do you monitor a system like that?

E.g. speech recognition is usually implemented with a pipilenie:

1- The user submits an audio
2- Audio is fed to a VAD module (Voice activity detection) that checks whether someone is speaking
3- If VAD says someone IS speaking, it sends it to the speech recognition system that generates a transcript.

Usually VAD is done by a learning algorithm, just like the speech recognition system. The later may run in the cloud, so we don’t want to send it non-speaking audios.

Cascading effects in the pipleine may be complex to track, involving different ML and non-ML components. You should monitor software, input and output metrics for each component and well as overall

### Practical assignment

(fill this in, it's interesting)

# Week 2 - Modelling

## A data centric approach

Let's delve into the modelling side of the ML cycle. The author insists on taking a **data centric AI approach**, where we focus on getting better data rather than the latest cutting edge model ever. Often any open source model would do, and it’s the data that is fed what makes the difference.

**AI sytem = Code + Data**

People tend to emphasize in the code, but for many (most) applications the model is a "solved" problem while the data is the variable part.

Remember that developing a model is an iterative process:
![[Pasted image 20241018062607.png]]

## Milestones during modelling

During the modelling phase, there are three increasingly complex **milestones** one has to achieve:

1. Do well on the training set. This is a necessary milestone.
2. Do well on the dev/test sets
3. Do well on business metrics / project goals.

![[Pasted image 20241018172809.png]]

Often the last 2 are not the same. One has to deal and with the use case in detail to find this out, e.g.:
- There may disproportionately important input sets for which the algorithm just cannot do wrong, such as
    - Web search: we cannot miss “navigational” queries like someone searching for youtube or reddit. You got to return the exact result else the users will lose confidence.
- Bias/fairness
    - ML for loan approval: make sure not to discriminate by protected attributes like gender
    - Product recommendations from retailers: treat major an minor retailers fairly.
- Rare cases. We may have a skewed data distribution, and we may care about being accurate in rare classes.
    - Medical diagnose.

![[Pasted image 20241018062857.png]]
## Defining a baseline

To tackle these challenges, it is important to establish a **baseline** on the level of **performance expected**. For example, on speech recognition, we may have certain groups of inputs and we may use human performance as a baseline to know how much we can aim to improve on each subset:

![[Pasted image 20241018062752.png]]

How to get a baseline:
- Humans do very well on unstructured data (image audio text) , bad in structure (tabular) data. Human level performance (HLP) is a common baseline
- Literature search for state-of-the-art/open source
- An older system
Baseline gives an estimate of the irreducible error / Bayes error and indicates what might be possible. 

General rule:
* For unstructured data (images, audio, etc), HLP is good because humans do great
* structured data (tables with numbers, basically): HLP may not be a good choice

## Tips 

- If you’re about to get started with a ML project modelling
    - Do literature search to see what’s possible
    - Find an open-source implementation if available
    - **A reasonable algorithm with good data will often outperform a great algorithm with not so good data. Care more about data than algorithms.**
    - Take into account deployment constraints if appropriate. Maybe not if you’re just doing a proof of concept
- Sanity check for code and algorithm
    - Try to overfit a small training dataset (maybe just a sample) before training on a large dataset.
- calculate performance on key slices of the dataset
- consider deployment constraints when picking a model


## Error analysis and prioritising what to work on 

Manually with a simple spreadsheet go over your dataset, see failures, identify groups or patterns that may be mispredicted, tag them.

![[Pasted image 20241018063024.png]]

The goal is to identify categories where you could productively improve the algorithm. Check for each tag (category) metrics such as:

- Fraction of total erros with this tag
- Fraction of misclassifications within that tag
- What fraction of all data has this tag?
- How much room for improvement is there on data with that tag? E.g. vs human leven performance.

This can help you decide to **priorize what to work on.** More on that:

![[Pasted image 20241018063101.png]]
If we improve performance on clean speech by 1%, given that 60% of the set is clean speech, we’ll raise average accuraby by 0.6%, etc etc. So bear in mid that both the gap of potential improvement and the % of data of each class is relevant.

Even though there is so much room for improvement in car noise, there’s larger potential for improvement in clean speech and people noise.
Other factors:

- How easy it is to improve accuracy in that category
- How important it is to improve in that category. E.g. car noise may be super important for hands off searches.

In order to improve specific categories you may:
- Collect more data
- Use data augmentation
- Improve label accuracy / data quality

Collecting data is expensive, so it’s important to make an analysis of what categories you want to focus.

## Skewed datasets

There may be an imbalance in the labels. E.g. find defects in manufacturing ; medical diagnosis (99% don;’t have a disease)

In skewed datasets, raw accuracy is not good. Just a constant negative classifier has super high classification. Instead, use a confusion matrix and metrics derived from it:

Precision = TP/(FP+FP)
Recall = TP/(TP+FN)

They can be combined in **F1 score** if you want an algorithm that does well in both. It goes down significantly if either is very small.

F1 = 2 / (1/P + 1/R

In the context of multi-class this generalizes easily:

![[Pasted image 20241018063327.png]]

Instead of accuracy, you may use F1 and decide from here to improve your model in “Discoloration”.

## Performance auditing

Once you feel the model is done, it’s good to audit its performance one last time.

![[Pasted image 20241018063415.png]]
There are MLOps tools to do this automatically. E.g. tensorflow has a package for model analysis TFMA that computes metrics on different slices of data.

Example:
![[Pasted image 20241018063455.png]]

## Experiment tracking

Having a system for tracking many experiments is crucial.
![[Pasted image 20241018063509.png]]Big teams / complex experiments may require an experiment tracking system such as Weigths and Biases.

![[Pasted image 20241018063519.png]]**TL;DR : Do track your experiments!**
# Week 3 - Data

## Data

### Data Definiton

![[Pasted image 20241018063550.png]]

Make sure there is no ambiguity in the labels. Reach an agreement between labellers or adjust labels if they come from different sources.

Data definition questions:
- What is the input x: features if it's structured data, pictures (maybe with a specific lighting / trait), etc
- What is the output y 

### Types of data problems

The best practices for organizing data (according to Andrew) depend on which of the 4 categories the problem falls in:

![[Pasted image 20241018220213.png]]

* Humans are great at processing unstructured data, but not that good at processing structured data.
* **Augmentation**: In unstructured data, data augmentation can be used to generate more data. In contrast, for structured data, it's hard to synthesize new data. It may also be harder to make humans label more data.
* **Label quality**: If the dataset is small, having clean labels is critical, and it's advisable to manually inspect them. If the dataset is huge (1M) it may be harder or impossible to manually inspect labels, and it's worth emphasising on the data process so that collection and instructions for labelling are clear.

### Label consistency
E.g. phone defect detection in a factory with picture. The size of the scratch may be confusing: do we need to label big boxes, not boxes at all? Solutions

- Average responses by many annotations to increase accuracy
- Ask the annotators to sit down and reach an agreement e.g. min 0.3mm to be labelled
- When there is disagreement, have a subject expert discuss a definition.
- (other problems) have a class or label to capture uncertainty, e.g. speech recognition the transcript could be "go to the \[unintelligible\]"
- in small data scenarios, labelers can actually go and discuss specific labels
- in big data scenarios, get a consistent definition with a small group, then send labelling instructions 

Problems with a large dataset may have a long tail of rare events in the input. Rare occurences that are critical to get right e.g. self driving car kid running in the highway. Label consistency for those instances is critical as well.

### HLP

We can use HLP to estimate the bayes irreducible error and do error analysis and prioritization.

When the ground truth label is **eternally defined**, HLP gives an estimate for Bayers / irreducible error.

When the label y comes from a human label, `HLP << 100%` may indicate ambiguous labelling instruction.

### Obtaining data
How much time should you spend obtaining data? Andrew suggests starting experiments asap instead of waiting to have the full sized dataset collected. Then iterate doing error analysis to guide what extra data you need.

Make an inventory of where you can get data adn what is it you already have

![[Pasted image 20241018222630.png]]
![[Pasted image 20241018222711.png]]

### Misc

![[Pasted image 20241018222751.png]]

Train-dev-test: 60/20/20
you may craft the division or even the batches to ensure certain slices are evenly represented.

## Scoping
Let's use the example of an e commerce retailer looking to increase sales. If you were to sit down and brainstorm, you might come up with many ideas, such as maybe a better product recommendation system, or a better search so people can find what they're looking for, or you may want to improve
quality of the catalogue data.

![[Pasted image 20241018223156.png]]

![[Pasted image 20241018223227.png]]get together with a business or private owner (someone that understands
the business) and brainstorm with them. What are their business or application problems?

In order to assess technical feasibility: 
![[Pasted image 20241018223542.png]]

* HLP: Can a human given the same data perform the task?
* Is this a new problem or has it already been tackled?
* History of project: make a plot time vs error and see historical projects where they stood.

**Diligence on value**
This is about having technical (MLE) and business teams to agree on metrics that both are comfortable with. There is usually a gap here that needs to be bridged.

**Milestones and Resourcing**
The following need to be estimated

![[Pasted image 20241018224303.png]]
