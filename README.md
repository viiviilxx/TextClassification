# CNN for Sentence Classification 
Implementation of [Convolutional Neural Networks for Sentence Classification](https://www.aclweb.org/anthology/D14-1181/) using PyTorch.
And using [Bidirectional Encoder Representations from Transformers](https://arxiv.org/abs/1810.04805) instead of word2vec for embedding word.

> Yoon Kim, Convolutional Neural Networks for Sentence Classification. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1746 - 1751, 2014.

# Requirements
- Python: 3.8.0 or higher
- PyTorch: 1.6.0 or higher
- Optuna: 2.0.0 or higher
- Transformers from haggingface 3.0.2 or higher

If you installed Anaconda, you can create a virtual enviroment from `env.yml`.
```
$ conda env create -f env,yml
```

# Datasets
If you use dataset prepared, download [RCV1-ids](https://drive.google.com/file/d/1kBKbH2sOjHZc-jJgayFO5FP8dK8tMrgk/view?usp=sharing) and put a unpacking folder into data/.

This is RCV1 dataset that raw text converted to ids by `BERT tokenizer`.
Raw text means it didn`t normalize.
Embedding`s Max length is 512 by BERT-base. So, used only 512 words from the beggining of each texts.
This dataset splitted 23,149 training sample set and 781,265 testing sample set according to [RCV1: A New Benchmark Collection for Text Categorization Research](https://www.jmlr.org/papers/volume5/lewis04a/lewis04a.pdf). 

If you want to use original or another datasets, you should convert datasets to ids by `BERT tokenizer` and you should change format.
```
- one document per line.
- line must has tokenized ids, label that represented in one hot vector.
```
And you should fix some parameter in run.py such as `classes`.

example
```
[id1, id2, id3, ...]<TAB>[000100...]
```

# BERT
This program embeds word into 768-dimensional vector by BERT.
This program use BERT of `base-uncased`

# Evalution Metrics
This program use Precision@k, MicroF1 and MacroF1.

# How to run
## normal training and testing
```
$ python run.py
```

## parameter search
```
$ python run.py --tuning
```

## Force to use cpu
```
$ python run.py --no_cuda
```

# Acknowledgment
This program is based on the following repositories.
Thank you very much for their accomplishments.

- [siddsax/XML-CNN](https://github.com/siddsax/XML-CNN) (MIT license)