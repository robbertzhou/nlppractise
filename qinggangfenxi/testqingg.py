#coding : utf-8

# 导包
import re
import os

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jieba
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split

# 使用gensim加载预训练中文分词，需要等待一段时间
cn_model = KeyedVectors.load_word2vec_format(r'G:\testdata\sgns.zhihu.bigram',
                                            binary=False, unicode_errors='ignore')
print('success...')

## 4.读取训练数据
pos_file_list = os.listdir(r'G:/testdata/train_txt/pos')
neg_file_list = os.listdir(r'G:/testdata/train_txt/neg')
pos_file_list = [r'G:/testdata/train_txt/pos/{}'.format(x) for x in pos_file_list]
neg_file_list = [r'G:/testdata/train_txt/neg/{}'.format(x) for x in neg_file_list]
pos_neg_file_list = pos_file_list + neg_file_list
# 读取所有的文本，放入到x_train,前3000是正向样本，后3000负向样本
x_train = []
for file in pos_neg_file_list:
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read().strip()
        pass
    x_train.append(text)
    pass

x_train = np.array(x_train)
y_train = np.concatenate((np.ones(3000), np.zeros(3000)))  # 生成标签

# 打乱训练样本和标签的顺序
np.random.seed(116)
np.random.shuffle(x_train)

np.random.seed(116)
np.random.shuffle(y_train)

x_train_tokens = []
for text in x_train:
    # 使用jieba进行分词
    cut = jieba.cut(text)
    cut_list = [x for x in cut]
    for i,word in enumerate(cut_list):
        try:
            # 将词转换为索引index
            cut_list[i] = cn_model.vocab[word].index
            pass
        except KeyError:
            # 如果词不在字典中，则输出0
            cut_list[i] = 0
            pass
        pass
    x_train_tokens.append(cut_list)
    pass

# 获取每段语句的长度，并画图展示
tokens_count = [len(tokens) for tokens in x_train_tokens]
tokens_count.sort(reverse=True)
# 画图查看词的长度分布
plt.plot(tokens_count)
plt.ylabel('tokens count')
plt.xlabel('tokens length')
plt.show()
# 可以看出大部分词的长度都是在500以下的
tokens_length = np.mean(tokens_count) + 2 * np.std(tokens_count)
print(tokens_length)

np.sum(tokens_count < tokens_length) / len(tokens_count)

# 定义一个把tokens转换成文本的方法
def reverse_tokens(tokens):
    text = ''
    for index in tokens:
        if index != 0:
            text = text + cn_model.index2word[index]
        else:
            text = text + ''
        pass
    return text
# 测试
print(reverse_tokens(x_train_tokens[0]))
print(y_train[0])

embedding_matrix = np.zeros((50000, 300))
for i in range(50000):
    embedding_matrix[i, :] = cn_model[cn_model.index2word[i]]
    pass
embedding_matrix = embedding_matrix.astype('float32')
# 检查index是否对应
# 输出300意义为长度为300的embedding向量一一对应
print(np.sum(cn_model[cn_model.index2word[300]] == embedding_matrix[300]))


x_train_tokens_pad = tf.keras.preprocessing.sequence.pad_sequences(x_train_tokens,
                                                                  maxlen=int(tokens_length),
                                                                  padding='pre',
                                                                  truncating='pre')
# 超出五万个词向量的词用0代替
x_train_tokens_pad[x_train_tokens_pad >= 300] = 0

# x_tokens_train, x_tokens_test, y_tokens_train, y_tokens_test = train_test_split(
#     x_train_tokens_pad,
#     y_train,
#     test_size=0.1,
#     random_state=12
# )

x_tokens_train = x_train_tokens_pad[:-int(x_train_tokens_pad.shape[0] / 10)]
x_tokens_test = x_train_tokens_pad[-int(x_train_tokens_pad.shape[0] / 10):]
y_tokens_train = y_train[:-int(y_train.shape[0] / 10)]
y_tokens_test = y_train[-int(y_train.shape[0] / 10):]

# 构建模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(50000,300,
                                         weights=[embedding_matrix],
                                         input_length=int(tokens_length),
                                         trainable=False
                                        ),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=True)),
    tf.keras.layers.LSTM(16, return_sequences=False),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy']
             )

# 训练20轮，每轮都进行测试集的验证，使用1%用来测试集，每批128
history = model.fit(x_tokens_train,
          y_tokens_train,
          batch_size=10,
          epochs=20,
          validation_split=0.1,
          validation_freq=1
         )


model.save('the_save_model.h5')