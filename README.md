# MultiSentiNet
MultiSentiNet: A Deep Semantic Network for Multimodal Sentiment Analysis-CIKM2017 

# Data
The datasets MVSA-Single and MVSA-Multi are easy to find from the citation in our paper. For the MSVA-Multi dataset we firstly get the real label for single modality by taking the majority vote out of the 3 sentiments; that is, an image or a text is considered valid only when at least 2 of the 3 annotators agree on the exact label. It is very natural to classify the samples when the textual label is consistent with the visual label. However, there are many tweets, in which the labels of text and image are inconsistent. So we denote some judgement rules to handle this problem in the following table.

|     | Number of labels of text and image |     |    Ground    |
|:----------------------------------:|:---:|:---:|:------------:||                 Pos                | Neu | Neg |     Truth    ||                  2                 |  0  |  0  |      Pos     ||                  1                 |  1  |  0  |      Pos     ||                  0                 |  1  |  1  |      Neg     ||                  0                 |  0  |  2  |      Neg     ||                  0                 |  2  |  0  |      Neu     ||                  1                 |  0  |  1  |   (Remove）  |


# Cite
If you use this code, please cite the following our paper:

Nan Xu and Wenji Mao. 2017. MultiSentiNet: A Deep Semantic Network for Multimodal Sentiment Analysis. In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management (CIKM '17). [[pdf](http://delivery.acm.org/10.1145/3140000/3133142/p2399-xu.pdf?ip=159.226.177.200&id=3133142&acc=ACTIVE%20SERVICE&key=33E289E220520BFB%2E949A0B1AADF887FF%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&CFID=1025106513&CFTOKEN=25773156&__acm__=1515483274_72ab3affea6608ade35350763de3da7f)]

