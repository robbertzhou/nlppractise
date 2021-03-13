import numpy as np


def preprocess(text):
    text = text.lower()
    words = text.split(' ')
    id_to_word = {}
    word_to_id = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    corpus = [word_to_id[word] for word in words]
    corpus = np.array(corpus)
    return corpus,word_to_id,id_to_word

def create_to_matrix(corpus,vocab_size,window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size,vocab_size),dtype=np.int32)

    for idx,word_id in enumerate(corpus):
        for i in range(1,window_size+1):
            left_idx = idx -i
            right_idx = idx + i
            if left_idx >= 0:
                left_word_idx = corpus[left_idx]
                co_matrix[word_id,left_word_idx] += 1

            if right_idx < corpus_size:
                right_word_idx = corpus[right_idx]
                co_matrix[word_id,right_word_idx] = 1
    return co_matrix

def cos_similarity(x,y):
    nx = x / np.sqrt(np.sum(x**2))
    ny = y / np.sqrt(np.sum(y**2))
    return np.dot(nx,ny)

x = np.array([1,1])
y = np.array([1,1])
print(np.dot(x,y))

text = 'You say goodbye and I say hello .'
text = text.lower()
print(text)
words = text.split(' ')
print(type(words))

id_to_word = {}
word_to_id = {}
for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word


print(id_to_word)
print(word_to_id)



corpus = [word_to_id[word] for word in words]
corpus = np.array(corpus)
print(corpus)

corpus,word_to_id,id_to_word = preprocess('You say goodbye and I say hello .')
print(corpus)
print(word_to_id)
print(id_to_word)

print(create_to_matrix(corpus,7))

