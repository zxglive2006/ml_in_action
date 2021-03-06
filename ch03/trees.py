# coding=utf-8
"""
Decision Tree Source Code for Machine Learning in Action Ch. 3
"""
from math import log
import operator


def create_data_set():
    _data_set = [[1, 1, 'yes'],
                 [1, 1, 'yes'],
                 [1, 0, 'no'],
                 [0, 1, 'no'],
                 [0, 1, 'no']]
    _labels = ['no surfacing', 'flippers']
    # change to discrete values
    return _data_set, _labels


def calc_shannon_entropy(_data_set):
    num_entries = len(_data_set)
    label_counts = {}
    for featVec in _data_set:     # the the number of unique elements and their occurance
        current_label = featVec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_entropy = 0.0
    for key in label_counts:
        prob = float(label_counts[key])/num_entries
        shannon_entropy -= prob * log(prob, 2)    # log base 2
    return shannon_entropy


def split_data_set(_data_set, axis, value):
    ret_data_set = []
    for featVec in _data_set:
        if featVec[axis] == value:
            reduced_feat_vec = featVec[:axis]     # chop out axis used for splitting
            reduced_feat_vec.extend(featVec[axis+1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


def choose_best_feature_to_split(_data_set):
    # the last column is used for the _labels
    num_features = len(_data_set[0]) - 1
    base_entropy = calc_shannon_entropy(_data_set)
    best_info_gain = 0.0
    best_feature = -1
    # iterate over all the features
    for i in range(num_features):
        # create a list of all the examples of this feature
        feat_list = [example[i] for example in _data_set]
        # get a set of unique values
        unique_vals = set(feat_list)
        new_entropy = 0.0
        for value in unique_vals:
            sub_data_set = split_data_set(_data_set, i, value)
            prob = len(sub_data_set)/float(len(_data_set))
            new_entropy += prob * calc_shannon_entropy(sub_data_set)
        # calculate the info gain; ie reduction in entropy
        info_gain = base_entropy - new_entropy
        # compare this to the best gain so far, if better than current best, set to best
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    # returns an integer
    return best_feature


def majority_cnt(class_list):
    """
    采用多数表决的方法决定该叶子节点的分类
    :param class_list:
    :return:
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(_data_set, _labels):
    """
    创建决策树
    :param _data_set: 数据集
    :param _labels: 标签列表
    :return:
    """
    class_list = [example[-1] for example in _data_set]
    if class_list.count(class_list[0]) == len(class_list):
        # stop splitting when all of the classes are equal
        return class_list[0]
    if len(_data_set[0]) == 1:
        # stop splitting when there are no more features in _data_set
        return majority_cnt(class_list)
    best_feat = choose_best_feature_to_split(_data_set)
    best_feat_label = _labels[best_feat]
    my_tree = {best_feat_label: {}}
    del(_labels[best_feat])
    feat_values = [example[best_feat] for example in _data_set]
    unique_vals = set(feat_values)
    for value in unique_vals:
        # copy all of _labels, so trees don't mess up existing _labels
        sub_labels = _labels[:]
        my_tree[best_feat_label][value] = create_tree(
            split_data_set(_data_set, best_feat, value), sub_labels)
    return my_tree


def classify(input_tree, feat_labels, test_vec):
    """
    使用决策树的分类函数
    :param input_tree:
    :param feat_labels:
    :param test_vec:
    :return:
    """
    first_str = list(input_tree.keys())[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    key = test_vec[feat_index]
    value_of_feat = second_dict[key]
    if isinstance(value_of_feat, dict):
        class_label = classify(value_of_feat, feat_labels, test_vec)
    else:
        class_label = value_of_feat
    return class_label


def store_tree(_input_tree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(_input_tree, fw)
    fw.close()


def grab_tree(filename):
    import pickle
    fr = open(filename, "rb")
    return pickle.load(fr)
    

if __name__ == '__main__':
    # myDat, labels = create_data_set()
    # print(labels)
    # print(split_data_set(myDat, 0, 1))
    # print(split_data_set(myDat, 0, 0))
    # print(choose_best_feature_to_split(myDat))
    # from ch03.treePlotter import retrieve_tree
    # my_tree = retrieve_tree(0)
    # print(my_tree)
    # print(classify(my_tree, labels, [1, 0]))
    # print(classify(my_tree, labels, [1, 1]))
    # store_tree(my_tree, 'classifierStorage.txt')
    # my_tree = grab_tree("classifierStorage.txt")
    # print(my_tree)
    fr = open("lenses.txt")
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    print(lenses[:3])
    lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenses_tree = create_tree(lenses, lenses_labels)
    print(lenses_tree)
    from ch03.treePlotter import create_plot
    create_plot(lenses_tree)
    print("Run Decision Tree finish")
