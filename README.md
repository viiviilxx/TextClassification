# CNN for Sentence Classification 
Implementation of [Convolutional Neural Networks for Sentence Classification](https://www.aclweb.org/anthology/D14-1181/) using PyTorch.
And using [Bidirectional Encoder Representations from Transformers](https://arxiv.org/abs/1810.04805) instead of word2vec for embedding words.

> Yoon Kim, Convolutional Neural Networks for Sentence Classification. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1746 - 1751, 2014.

# Requirements
- Python: 3.8.0 or higher
- PyTorch: 1.6.0 or higher
- Optuna: 2.0.0 or higher
- Transformers from huggingface 3.0.2 or higher

If you installed Anaconda, you can create a virtual enviroment from `env.yml`.
```
$ conda env create -f env.yml
```

# Datasets
If you use sample dataset, download [RCV1-ids](https://drive.google.com/file/d/1kBKbH2sOjHZc-jJgayFO5FP8dK8tMrgk/view?usp=sharing) and put the folder that unzipped the `id.zip` into data/.

This is RCV1 dataset that raw texts converted to ids by `BERT tokenizer`.
Raw texts means it didn\`t normalize.
Embedding\`s Max length is 512 by BERT-base. So, used only 512 words from the beggining of each texts.
This dataset splitted 23,149 training sample and 781,265 testing sample according to [RCV1: A New Benchmark Collection for Text Categorization Research](https://www.jmlr.org/papers/volume5/lewis04a/lewis04a.pdf). 

If you want to use original or another dataset, you should convert dataset to ids by `BERT tokenizer` and you should change format.
```
- one document per line
- line must has 'tokenized ids' and 'label'
- label must be represented in one hot vector
```
example
```
[id1, id2, id3, ...]<TAB>[000100...]
```
And you should fix some parameters in run.py such as `classes` and `embedding_dim`.


# BERT
[huggingface](https://github.com/huggingface/transformers) publish some pre-trained models including BERT.
This program uses `base-uncased` model that one of pre-trained models.
So, embeds a word into a 768-dimensional vector by BERT.
And this program turn off the fine-tuning of BERT.

# Evalution Metrics
This program uses Precision@k, MicroF1 and MacroF1.

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
- [yu54ku/xml-cnn](https://github.com/yu54ku/xml-cnn) (MIT license)