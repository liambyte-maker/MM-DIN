# M3Rec: Multi-Modal Learning for Cold-Start CTR Recommendation

> Research Project | Cold-Start Recommendation · CTR Prediction · Multimodal Learning

## Overview

This project explores how multimodal information can improve cold-start recommendation performance in industrial CTR ranking systems.

Modern CTR models such as DIN, DMT, DeepFM, and DCN heavily rely on Item ID Embeddings. While these embeddings work well for popular items with abundant interactions, they often perform poorly for newly launched and long-tail items due to sparse training signals.

To address this issue, this project investigates multiple approaches for incorporating multimodal representations into DIN-based ranking models and studies the interaction between ID features and multimodal features.

---

## Motivation

Most industrial CTR models follow a simple paradigm:

```text
Embedding
     +
MLP
     ↓
CTR Prediction
```

Among all features, Item ID Embeddings contribute the strongest predictive power because they directly capture user-item interaction signals.

However:

```text
Popular Item
→ Well-trained ID Embedding

Cold-Start Item
→ Poorly-trained ID Embedding
```

This motivates the use of multimodal information such as:

* Image Features
* Text Features
* Category Features
* Brand Features

to improve representation quality for cold-start items.

---



## Research Roadmap

## base Model

<img width="1529" height="1132" alt="image" src="https://github.com/user-attachments/assets/c86cbaa5-9383-41f0-9c72-42f253edc6af" />



### Approach 1: Cluster Attention

Cluster multimodal item embeddings into semantic groups.

```text
Item Embedding
      ↓
  Clustering
      ↓
  K Clusters
```

A cluster-level behavior sequence is added to DIN, allowing the model to learn both:

* Item-level interests
* Category-level interests

---

### Approach 2: Similarity Distribution Modeling

Compute cosine similarities between the target item and historical items:

$$
\text{Sim}_i =
\frac{d_t^{T} d_i}
{|d_t| \cdot |d_i|}
$$

Construct:

```text
S = [Sim₁, Sim₂, ..., Sim₅₀]
```

and feed similarity features into the ranking model.

To improve robustness, DeepSets is introduced to model similarity distributions rather than individual similarity values.

---

### Approach 3: Two-Stage Multimodal Training

<img width="1032" height="777" alt="image" src="https://github.com/user-attachments/assets/8d9128ac-3af4-46ab-8d9c-242418dc35f0" />


Directly combining:

```text
ID Embedding
+
Multi-Modal Embedding
```

often causes a phenomenon called:

```text
ID Dominance
```

where the model relies heavily on ID features while ignoring multimodal representations.

To mitigate this issue:

1. Pre-train multimodal branches using CTR/CVR objectives.
2. Freeze multimodal encoders.
3. Jointly train with the main DIN model.

This allows multimodal representations to better align with ranking tasks before integration.

---

## Key Findings

### Finding 1: Cold-Start Limitation of ID Embeddings

Popular items receive sufficient updates, while cold-start items often suffer from under-trained embeddings.

### Finding 2: Raw Multimodal Embeddings Do Not Always Help

Experimental results showed:

```text
Cosine Similarity Features
>
Raw Multi-Modal Embeddings
```

indicating that representation utilization is often more important than representation complexity.

### Finding 3: ID Dominance

ID features learn significantly faster than multimodal features:

```text
ID Features
→ Near convergence in one epoch

Multi-Modal Features
→ Require multiple epochs
```

This creates optimization imbalance during joint training.

### Finding 4: Similarity Modeling Improves Stability

Transforming multimodal representations into similarity features reduces optimization difficulty and improves cold-start robustness.

---


## Contributions

* Built a DIN-based multimodal CTR ranking framework.
* Proposed Cluster Attention for category-level interest modeling.
* Proposed Similarity Distribution Modeling with DeepSets.
* Identified the ID Dominance phenomenon in multimodal recommendation systems.
* Designed a Two-Stage Multimodal Training framework for cold-start recommendation.

---

## Tech Stack

* Python
* PyTorch
* DIN
* DeepSets
* CLIP
* BERT
* CTR Prediction
* Recommendation Systems

---

## Future Work

* [ ] SASRec Integration
* [ ] Multimodal Sequential Recommendation
* [ ] Confidence-Aware Fusion
* [ ] CLIP-Enhanced Recommendation
* [ ] LLM Recommendation
* [ ] Agent Recommendation Systems

---

## Author

**Jinming Liu**

### Research Interests

* Recommendation Systems
* CTR Prediction
* Cold-Start Recommendation
* Multimodal Learning
* Sequential Recommendation
