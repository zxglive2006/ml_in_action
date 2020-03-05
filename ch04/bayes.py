# coding=utf-8
"""
Created on Oct 19, 2010
@author: Peter
NASA RSS Feeds：https://www.nasa.gov/content/nasa-rss-feeds
停用词表：https://www.ranks.nl/stopwords
"""
from numpy import *


def load_data_set():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]    # 1 is abusive, 0 not
    return posting_list, class_vec


def create_vocab_list(_data_set):
    vocab_set = set([])  # create empty set
    for document in _data_set:
        vocab_set = vocab_set | set(document)   # union of the two sets
    return list(vocab_set)


def set_of_words_to_vec(vocab_list, input_set):
    return_vec = [0]*len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return return_vec


def train_nb0(train_matrix, train_category):
    """
    朴素贝叶斯分类器训练函数
    :param train_matrix: 文档矩阵
    :param train_category: 每篇文档类别标签所构成的向量
    :return:每个类别的条件概率和侮辱性文档出现的先验概率
    """
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_train_docs)
    p0_num = ones(num_words)
    p1_num = ones(num_words)      # change to ones()
    p0_denom = 2.0      # 分母denominator
    p1_denom = 2.0                # change to 2.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p1_vect = log(p1_num/p1_denom)          # change to log(), 自然对数
    p0_vect = log(p0_num/p0_denom)          # change to log()
    return p0_vect, p1_vect, p_abusive


def classify_nb(vec2_classify, p0_vec, p1_vec, p_class1):
    p1 = sum(vec2_classify * p1_vec) + log(p_class1)    # element-wise multiply
    p0 = sum(vec2_classify * p0_vec) + log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else: 
        return 0


def bag_of_words_to_vec(vocab_list, input_set):
    return_vec = [0]*len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec


def testing_nb():
    list_o_posts, list_classes = load_data_set()
    my_vocab_list = create_vocab_list(list_o_posts)
    train_mat = []
    for post_in_doc in list_o_posts:
        train_mat.append(set_of_words_to_vec(my_vocab_list, post_in_doc))
    p0_v, p1_v, p_ab = train_nb0(array(train_mat), array(list_classes))
    test_entry = ['love', 'my', 'dalmation']
    this_doc = array(set_of_words_to_vec(my_vocab_list, test_entry))
    print(test_entry, 'classified as: ', classify_nb(this_doc, p0_v, p1_v, p_ab))
    test_entry = ['stupid', 'garbage']
    this_doc = array(set_of_words_to_vec(my_vocab_list, test_entry))
    print(test_entry, 'classified as: ', classify_nb(this_doc, p0_v, p1_v, p_ab))


def text_parse(big_string):
    """
    文本分解
    :param big_string: 文本字符串
    :return: word list
    """
    import re
    list_of_tokens = re.split(r'\W+', big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


def spam_test():
    doc_list = []
    class_list = []
    full_text = []
    for i in range(1, 26):
        word_list = text_parse(open('email/spam/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list = text_parse(open('email/ham/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = create_vocab_list(doc_list)      # create vocabulary
    total_size = 50
    test_set_size = int(0.2 * total_size)
    print("total size:{}, test set size:{}".format(total_size, test_set_size))
    training_set = list(range(total_size))
    test_set = []         # create test set
    for i in range(test_set_size):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del training_set[rand_index]
    train_mat = []
    train_classes = []
    for doc_index in training_set:
        train_mat.append(bag_of_words_to_vec(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    # train the classifier (get probability) trainNB0
    p0_v, p1_v, p_spam = train_nb0(array(train_mat), array(train_classes))
    error_count = 0
    for doc_index in test_set:        # classify the remaining items
        word_vector = bag_of_words_to_vec(vocab_list, doc_list[doc_index])
        if classify_nb(array(word_vector), p0_v, p1_v, p_spam) != class_list[doc_index]:
            error_count += 1
            print("classification error: ", doc_list[doc_index])
    print('the error rate is: ', float(error_count)/len(test_set))
    return vocab_list, full_text


def calc_most_freq(vocab_list, full_text, top_size=30):
    import operator
    freq_dict = {}
    for token in vocab_list:
        freq_dict[token] = full_text.count(token)
    sorted_freq = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_freq[:top_size]


def local_words(feed0, feed1):
    doc_list = []
    class_list = []
    full_text = []
    min_len = min(len(feed0['entries']), len(feed1['entries']))
    for i in range(min_len):
        word_list = text_parse(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
        word_list = text_parse(feed1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
    vocab_list = create_vocab_list(doc_list)  # create vocabulary
    top_words = calc_most_freq(vocab_list, full_text, 8)   # remove top words
    print("top_words:{}".format(top_words))
    for pairW in top_words:
        if pairW[0] in vocab_list:
            vocab_list.remove(pairW[0])
    total_size = 2 * min_len
    test_set_size = int(0.2 * total_size)
    print("total size:{}, test set size:{}".format(total_size, test_set_size))
    training_set = list(range(total_size))
    test_set = []           # create test set
    for i in range(test_set_size):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del training_set[rand_index]
    train_mat = []
    train_classes = []
    for doc_index in training_set:    # train the classifier (get probability) trainNB0
        train_mat.append(bag_of_words_to_vec(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    p0_v, p1_v, p_spam = train_nb0(array(train_mat), array(train_classes))
    error_count = 0
    for doc_index in test_set:        # classify the remaining items
        word_vector = bag_of_words_to_vec(vocab_list, doc_list[doc_index])
        if classify_nb(array(word_vector), p0_v, p1_v, p_spam) != class_list[doc_index]:
            error_count += 1
            print("classification error: ", doc_list[doc_index])
    print('the error rate is: ', round(float(error_count)/len(test_set), 2))
    return vocab_list, p0_v, p1_v


def get_top_words(feed0, feed1):
    vocab_list, p0_v, p1_v = local_words(feed0, feed1)
    top_feed0 = []
    top_feed1 = []
    threshold = -3.5
    for i in range(len(vocab_list)):
        if p0_v[i] > threshold:
            top_feed0.append((vocab_list[i], p0_v[i]))
        if p1_v[i] > threshold:
            top_feed1.append((vocab_list[i], p1_v[i]))
    sorted_feed0 = sorted(top_feed0, key=lambda pair: pair[1], reverse=True)
    print("feed0**feed0**feed0**feed0**feed0**feed0**feed0**feed0")
    for item in sorted_feed0:
        print(item[0])
    sorted_feed1 = sorted(top_feed1, key=lambda pair: pair[1], reverse=True)
    print("feed1**feed1**feed1**feed1**feed1**feed1**feed1**feed1")
    for item in sorted_feed1:
        print(item[0])


def feed_test():
    import feedparser
    # breaking_news = feedparser.parse('https://www.nasa.gov/rss/dyn/breaking_news.rss')
    # education_news = feedparser.parse("https://www.nasa.gov/rss/dyn/educationnews.rss")
    # vocab_list, p_breaking_news, p_education_news = local_words(breaking_news, education_news)
    local_breaking_news = feedparser.parse(r"./rss/breaking_news.rss")
    local_education_news = feedparser.parse(r"./rss/educationnews.rss")
    get_top_words(local_breaking_news, local_education_news)


if __name__ == '__main__':
    # testing_nb()
    # spam_test()
    feed_test()
    print("Run bayes finish")
