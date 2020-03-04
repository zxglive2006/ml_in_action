# coding=utf-8
"""
Created on Oct 19, 2010
@author: Peter
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
    p1_vect = log(p1_num/p1_denom)          # change to log()
    p0_vect = log(p0_num/p0_denom)          # change to log()
    return p0_vect, p1_vect, p_abusive


def classify_nb(vec2_classify, p0_vec, p1_vec, p_class1):
    p1 = sum(vec2_classify * p1_vec) + log(p_class1)    # element-wise multiply
    p0 = sum(vec2_classify * p0_vec) + log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else: 
        return 0


def bag_of_words_to_vec_mn(vocab_list, input_set):
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
    :param big_string: big string
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
    training_set = list(range(50))
    test_set = []         # create test set
    for i in range(10):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])
    train_mat = []
    train_classes = []
    for docIndex in training_set:    # train the classifier (get probs) trainNB0
        train_mat.append(bag_of_words_to_vec_mn(vocab_list, doc_list[docIndex]))
        train_classes.append(class_list[docIndex])
    p0_v, p1_v, p_spam = train_nb0(array(train_mat), array(train_classes))
    error_count = 0
    for docIndex in test_set:        # classify the remaining items
        word_vector = bag_of_words_to_vec_mn(vocab_list, doc_list[docIndex])
        if classify_nb(array(word_vector), p0_v, p1_v, p_spam) != class_list[docIndex]:
            error_count += 1
            print("classification error", doc_list[docIndex])
    print('the error rate is: ', float(error_count)/len(test_set))
    # return vocab_list, full_text


def calc_most_freq(vocab_list, full_text):
    import operator
    freq_dict = {}
    for token in vocab_list:
        freq_dict[token] = full_text.count(token)
    sorted_freq = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_freq[:30]


def local_words(feed1, feed0):
    doc_list = []
    class_list = []
    full_text = []
    min_len = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(min_len):
        word_list = text_parse(feed1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)    # NY is class 1
        word_list = text_parse(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = create_vocab_list(doc_list)  # create vocabulary
    top30_words = calc_most_freq(vocab_list, full_text)   # remove top 30 words
    for pairW in top30_words:
        if pairW[0] in vocab_list:
            vocab_list.remove(pairW[0])
    training_set = range(2*min_len)
    test_set = []           # create test set
    for i in range(20):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])
    train_mat = []
    train_classes = []
    for docIndex in training_set:    # train the classifier (get probs) trainNB0
        train_mat.append(bag_of_words_to_vec_mn(vocab_list, doc_list[docIndex]))
        train_classes.append(class_list[docIndex])
    p0_v, p1_v, p_spam = train_nb0(array(train_mat), array(train_classes))
    error_count = 0
    for docIndex in test_set:        # classify the remaining items
        word_vector = bag_of_words_to_vec_mn(vocab_list, doc_list[docIndex])
        if classify_nb(array(word_vector), p0_v, p1_v, p_spam) != class_list[docIndex]:
            error_count += 1
    print('the error rate is: ', float(error_count)/len(test_set))
    return vocab_list, p0_v, p1_v


def get_top_words(ny, sf):
    vocab_list, p0_v, p1_v = local_words(ny, sf)
    top_ny = []
    top_sf = []
    for i in range(len(p0_v)):
        if p0_v[i] > -6.0:
            top_sf.append((vocab_list[i], p0_v[i]))
        if p1_v[i] > -6.0:
            top_ny.append((vocab_list[i], p1_v[i]))
    sorted_sf = sorted(top_sf, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sorted_sf:
        print(item[0])
    sorted_ny = sorted(top_ny, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sorted_ny:
        print(item[0])


if __name__ == '__main__':
    # testing_nb()
    spam_test()
    print("Run bayes finish")
