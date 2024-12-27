---
tags:
  - Computer
  - Vision
aliases: 
publish: false
slug: recommender-system
title: collaborativ filtering
description: In this article I survey different backbone alternatives used in modern computer vision architectures
date: 2024-07-20
image: /thumbnails/backbones.png
---
**Data Collection**: The first step is to collect data about users and items. This data can include user interactions (e.g., clicks, purchases, ratings), item attributes (e.g., genre, price), and contextual information (e.g., time of day, location).

**Model Training**: Machine learning models are trained on the preprocessed data. These models learn patterns and relationships between users and items to make predictions about user preferences.

**Recommendation Generation**: Once the model is trained, it generates recommendations for users based on their past interactions and preferences. The recommendations can be personalized to each user, providing a tailored experience.



The primary problem is to accurately recommend items that a user is likely to enjoy or find useful.


- **Inputs**: User-item interaction data (e.g., ratings, clicks, purchases), user profiles (e.g., demographics, preferences), item attributes (e.g., genre, price), and contextual information (e.g., time of day, location).
    
- **Outputs**: A ranked list of recommended items for each user. Or a shore such as user's rating, click, or purchase probability for an item that they haven't interacted with yet.

# Collaborative filtering
Collaborative filtering is a classical approach where systems recommend items based on past interactions among users. It can be:

- **User-based**: Recommending items that similar users liked.
- **Item-based**: Recommending items similar to those a user has already interacted with.
Deep learning and transformer models are often applied to collaborative filtering to improve performance, particularly when dealing with large-scale, sparse data (many users, many items, but few interactions).

# Content-based filtering

The goal is to predict how much a user will like an item they have not interacted with yet, based on their past behavior and the behavior of similar users. The task is typically framed as a **regression** or **ranking problem**.

The main challenge is the **sparsity problem**: user-item interaction matrices are often sparse because users only interact with a small subset of items. Collaborative filtering aims to fill in the missing values in this matrix (i.e., predicting a user’s rating for an item they haven’t rated yet).

**User-Item Interaction Matrix**: This matrix represents the interactions between users and items. In traditional systems, this could be user ratings (e.g., a 1-5 star rating),

**Matrix Factorization**: The core idea is to approximate the user-item interaction matrix as the product of two lower-rank matrices:
- One matrix representing the **user features**.
- One matrix representing the **item features**.

A model could take as input:
* a list of user-item interactions
* the model learns a latent representation of users and of items
* the model predicts interaction scores bw items and users

TEAKEAWAY: Learn embeddings of users and of movies and the product gives you the ocmpatibility.
https://developers.google.com/machine-learning/recommendation/collaborative/matrix