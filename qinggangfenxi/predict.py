import tensorflow as tf
from tensorflow import keras
import jieba
from gensim.models import KeyedVectors

model = keras.models.load_model('the_save_model.h5')
cn_model = KeyedVectors.load_word2vec_format(r'G:\testdata\sgns.zhihu.bigram',
                                            binary=False, unicode_errors='ignore')

def predict_sentiment(text):
    print(text)
    # 分词
    cut = jieba.cut(text)
    cut_list = [x for x in cut]
    for i, word in enumerate(cut_list):
        try:
            cut_list[i] = cn_model.vocab[word].index
        except KeyError:
            cut_list[i] = 0
        pass
    # padding
    tokens_pad = tf.keras.preprocessing.sequence.pad_sequences([cut_list],
                                                               maxlen=int(300),
                                                               padding='pre',
                                                               truncating='pre')
    # 大于50000的归0，不归0模型的使用会报错
    tokens_pad[tokens_pad >= 50000] = 0
    return tokens_pad
    pass


test_list = [
    '酒店设施不是新的，服务态度很不好',
    '酒店卫生条件非常不好',
    '床铺非常舒适',
    '房间很冷，还不给开暖气',
    '房间很凉爽，空调冷气很足',
    '酒店环境不好，住宿体验很不好',
    '房间隔音不到位',
    '晚上回来发现没有打扫卫生,心情不好',
    '因为过节所以要我临时加钱，比团购的价格贵',
    '房间很温馨，前台服务很好,'
]
for text in test_list:
    try:
        tokens_pad = predict_sentiment(text)
        result = model.predict(x=tokens_pad)
        print(result)
        if result[0][0] <= result[0][1]:
            print(f'正:{result[0][1]}')
        else:
            print(f'负:{result[0][0]}')
    except Exception as ex:
        print(ex.args)
        pass
    pass