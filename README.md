# Fake News Detection

My solution to a DataCup competition [Leaders Prize: Fact or Fake News?](https://business.financialpost.com/pmn/press-releases-pmn/business-wire-news-releases-pmn/1m-leaders-prize-to-be-awarded-to-solution-combating-fake-news-using-artificial-intelligence).

## Problem Overview
The goal is to predict the truth ratings that human fact checkers would assign to each claim in the dataset based on some related articles and the metadata associated with each claim.

## Data
The dataset contains claims and the associated metadata from 9 fact checking websites. On those websites, professional fact checkers publish a truth rating for each claim with links to the related articles. The truth ratings provided were mapped to the labels:

- 0 (false)
- 1 (partly true)
- 2 (true)

## My solution

First, the claim and the related articles are preprocessed by converting each sentence into a TF-IDF representation. The 5 sentences that have the highest cosine similarity with a claim are extracted and concatenated with the metadata.

Then, Bi-directional Encoder Representations from Transformers (BERT) is fine-tuned based on these sentences and the metadata to predict the label of each claim.

See `BERT_claim_classification.ipynb`.

## Other attempts

- Used RNN to encode claim, metadata with a Feed Forward network on top to predict the labels. See `fake_news_detection_rnn.ipynb`.
- Used Transformer encoder (implemented from scratch) with a Feed Forward network to predict the labels. See `fake_news_detection_transformer.ipynb`.


Transformer implementation based on:
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [How to code the transformer in Pytorch blog post](https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec)
- [Attention is all you need explained](http://mlexplained.com/2017/12/29/attention-is-all-you-need-explained/)
