---
layout: post
title: Fake News Classification
---

In this tutorial I will be showing you guys how to develop and assess a fake news classifer using TensorFlow.

## Initialization

The data source we will be using comes from Kaggle. The following code will load in the neccessary packages along
with our data. We can also explore our dataset a bit as well.

```python
import tensorflow as tf
import numpy as np
import pandas as pd

train_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_train.csv?raw=true"
train_data = pd.read_csv(train_url)

train_data.head()
```
![alt text](https://i.gyazo.com/14fa6f886e8f0f37079ba265f1dd852f.png)

## Making our dataset

Now we'll make a TensorFlow dataset to hold our data. The dataset will have two inputs, the article text and title, along with one
output, whether its fake or not. First we'll have to remove 'stopwords' in our data such as 'the' or 'as' which don't really help
us in our analysis.

```python
import nltk
nltk.download('stopwords')
```