# Multi-Modal Enhanced DIN for Cold-Start Recommendation

## 项目简介

本项目围绕推荐系统中冷启动问题展开研究，探索如何利用多模态特征缓解传统ID Embedding在数据稀疏场景下训练不足的问题
当前工业界主流CTR 精排（DIN、DMT、DeepFM、DCN 等）高度依赖用户行为序列中的Item Id 建模用户兴趣
Item ID Embedding是最重要的特征来源，其优势在于表达能力强，参数量大，拟合能力优秀


对于热门商品
* 曝光充足
* 点击数据丰富
* Embedding 学习充分
  
然后对于新商品和长尾商品：

* 曝光次数少
* 点击数据不足
* ID Embedding训练不足

导致推荐效果下降

因此，本项目探索利用商品文本，图像等多模态特征增强目标商品表示，从而提升冷启动推荐能力。

## 数据分析

如何判断ID Embedding 是否学好？

||x||₂
训练次数越多，L2越大

| 训练阶段 | 无L2正则化 | L2正则化 |
| ---- | ------ | ----- |
| 初期   | 持续增长   | 持续增长  |
| 中期   | 继续增长   | 逐渐稳定  |
| 后期   | 继续增长   | 收敛稳定  |

如果高频商品的 Embedding Norm仍然增长，说明Embedding可能尚未充分收敛


# 研究路线

Base Model
<img width="1529" height="1132" alt="image" src="https://github.com/user-attachments/assets/eceebaea-05e3-474a-af69-82ce1982dd57" />


## 方案一： Cluster Attention

用全量商品item的多模态Embedding进行聚类，得到K个cluster，然后把cluset id 当特征输入到精排模型中，经过聚类后，每个item有一个cluser id,用target item的簇id和序列item cluster id 进行attention计算，用DIN 计算
相当于加一条“cluster序列”的计算

优势：
* 学习类别级兴趣
* 缓解冷启动问题

缺点
* 聚类需要定期更新
* 全量聚类成本高

## 方案二: Similarity Distribution Modeling


计算：

Target Item DMT Embedding

与

History Item DMT Embedding

之间的余弦相似度：
$\text{Sim}_i = \frac{d_t^{T} d_i}{\|d_t\| \cdot \|d_i\|}$

得到：

S=[Sim₁,...,Sim₅₀]

直接拼接到DNN底层。

## 方案三 原始多模态Embedding

直接把多模态Embedding 当DIN 输入
新增DMT DIN 分支，最后拼接

## 问题发现： ID Dominancee
方案二效果比方案三好

## 为什么直接使用多模态Embedding效果反而不好？

难点1： ID 特征学习速度远快于多模态特征
模型会优先SZ拟合：  Item ID

而忽略 Image Feature, Text Feature，最终导致 ID Dominace

难点2：
需要解决ID 表征（One -Epoch收敛)和多模态表征（Multi-Epoch收敛)的联合训练时因收敛速度不同，共同优化效果次优的问题



## 优化1： DeepSet Similarity Distribution

方案二直接拼接50维相似度序列，存在顺序敏感，维度过高，泛化不足的问题

DeepSet思想：把相似度看成一个集合，统计成20个桶的分布特征，从而消除顺序影响，降低维度，增强稳定性，本质是在学习用户历史与target的整体相似度分布，而不是某个位置上具体的相似度

## 优化2 Two-Stage Multimodal Training

解决ID学的太快，多模态学的太慢

第一阶段： 单独训练Multimodal DIN

<img width="1032" height="777" alt="image" src="https://github.com/user-attachments/assets/b691e829-e789-47df-86d9-f346177472dd" />


第二阶段： 把训练好的DIN Hidden Layer 迁移到正式推荐模型，实习多模态参数提前适配CTR任务，避免ID Dpminance。 这样就实现了多模态embedding提前训练多个epoch的目的


# 项目贡献（Contributions）
## Contribution 1

完成基于 DIN 的 CTR 精排框架搭建。

## Contribution 2

提出 Cluster Attention 方法，用于建模类别级用户兴趣。

## Contribution 3

提出 Similarity Distribution Modeling 方法，利用多模态语义信息增强冷启动推荐能力。

## Contribution 4

发现推荐系统中的 ID Dominance 现象：

ID特征学习速度>多模态特征学习速度

导致联合训练效果受限。

## Contribution 5

提出两阶段多模态训练框架（Two-Stage Training），缓解 ID 与多模态特征收敛速度不匹配的问题。

项目收获（Lessons Learned）

通过本项目，我对以下问题有了更加深入的理解：

DIN 与工业 CTR 模型结构；
* 冷启动推荐问题；
* 多模态表示学习；
* Attention 在推荐系统中的应用；
* Cosine Similarity 建模方法；
* DeepSet 集合表示学习；
I* D Dominance 现象及其成因；
* 多模态预训练与迁移学习。
  
## Future Work

未来计划进一步探索：

* SASRec 序列推荐模型；
* 多模态序列推荐；
* Confidence-Aware Fusion；
* CLIP 增强推荐；
* LLM Recommendation；
* Agent Recommendation Systems。
