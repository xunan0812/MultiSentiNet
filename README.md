# MultiSentiNet
MultiSentiNet: A Deep Semantic Network for Multimodal Sentiment Analysis-CIKM2017 

# Data
The datasets MVSA-Single and MVSA-Multi are easy to find from the citation in our paper or [[download](http://www.mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/)]. For the MSVA-Multi dataset we firstly get the real label for single modality by taking the majority vote out of the 3 sentiments; that is, an image or a text is considered valid only when at least 2 of the 3 annotators agree on the exact label. It is very natural to classify the samples when the textual label is consistent with the visual label. However, there are many tweets, in which the labels of text and image are inconsistent. So we denote some judgement rules to handle this problem in the following table.

<table class="tg">  <tr>    <th class="tg-baqh" colspan="3">Number of labels of text and image</th>    <th class="tg-baqh" rowspan="2">Ground<br>Truth</th>  </tr>  <tr>    <td class="tg-baqh">Pos</td>    <td class="tg-baqh">Neu</td>    <td class="tg-baqh">Neg</td>  </tr>  <tr>    <td class="tg-baqh">2</td>    <td class="tg-baqh">0</td>    <td class="tg-baqh">0</td>    <td class="tg-baqh">Pos</td>  </tr>  <tr>    <td class="tg-baqh">1</td>    <td class="tg-baqh">1</td>    <td class="tg-baqh">0</td>    <td class="tg-baqh">Pos</td>  </tr>  <tr>    <td class="tg-baqh">0</td>    <td class="tg-baqh">1</td>    <td class="tg-baqh">1</td>    <td class="tg-baqh">Neg</td>  </tr>  <tr>    <td class="tg-baqh">0</td>    <td class="tg-baqh">0</td>    <td class="tg-baqh">2</td>    <td class="tg-baqh">Neg</td>  </tr>  <tr>    <td class="tg-baqh">0</td>    <td class="tg-baqh">2</td>    <td class="tg-baqh">0</td>    <td class="tg-baqh">Neu</td>  </tr>  <tr>    <td class="tg-baqh">1</td>    <td class="tg-baqh">0</td>    <td class="tg-baqh">1</td>    <td class="tg-baqh">(Remove）</td>  </tr></table>


# Cite
If you use this code, please cite the following our paper:

Nan Xu and Wenji Mao. 2017. MultiSentiNet: A Deep Semantic Network for Multimodal Sentiment Analysis. In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management (CIKM '17). [[pdf](https://dl.acm.org/citation.cfm?id=3133142)]