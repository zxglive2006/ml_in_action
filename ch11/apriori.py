# encoding: UTF-8
"""
Created on Mar 24, 2011
Ch 11 code
@author: Peter
参考：
https://en.wikipedia.org/wiki/Apriori_algorithm
http://api.votesmart.org/docs/terms.html
https://github.com/ndanielsen/py-votesmart
https://archive.ics.uci.edu/ml/datasets/Mushroom
"""
from numpy import *
from time import sleep
from votesmart import VoteSmartAPI


def load_data_set():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def create_c1(data_set):
    """
    构建集合C1
    :param data_set:
    :return: 长度为1的所有候选项集的集合
    """
    c1 = []
    for transaction in data_set:
        for item in transaction:
            if not [item] in c1:
                c1.append([item])
    c1.sort()
    # use frozen set so we can use it as a key in a dict
    return list(map(frozenset, c1))


def scan_d(data_set, c_k, min_support):
    """
    从Ck生成Lk
    :param data_set: 数据集
    :param c_k: 候选项集列表
    :param min_support: 感兴趣项集的最小支持度
    :return: 频繁项集列表和包含支持度值的列表
    """
    ss_cnt = {}
    for tid in data_set:
        for can in c_k:
            if can.issubset(tid):
                if can not in ss_cnt:
                    ss_cnt[can] = 1
                else:
                    ss_cnt[can] += 1
    num_items = float(len(data_set))
    ret_list = []
    support_data = {}
    for _key in ss_cnt:
        support = ss_cnt[_key]/num_items
        if support >= min_support:
            ret_list.insert(0, _key)
        support_data[_key] = support
    return ret_list, support_data


def apriori_gen(item_set_k_1, k):
    """
    创建候选项集Ck
    :param item_set_k_1: 频繁项集元素列表Lk-1
    :param k: 候选项集元素个数
    :return: 候选项集列表Ck
    """
    candidate_set_k = []
    len_item_set_k_1 = len(item_set_k_1)
    for i in range(len_item_set_k_1):
        for j in range(i+1, len_item_set_k_1):
            lst_1 = list(item_set_k_1[i])[:k - 2]
            lst_2 = list(item_set_k_1[j])[:k - 2]
            lst_1.sort()
            lst_2.sort()
            if lst_1 == lst_2:  # if first k-2 elements are equal
                candidate_set_k.append(item_set_k_1[i] | item_set_k_1[j])   # set union
    return candidate_set_k


def apriori(data_list, min_support=0.5):
    """
    发现频繁项集
    :param data_list: 原始数据列表
    :param min_support: 最小支持度阈值
    :return: 频繁项集列表和包含支持度数据的频繁项集字典
    """
    candidate_set_1 = create_c1(data_list)
    data_set = list(map(set, data_list))
    item_set_1, support_data = scan_d(data_set, candidate_set_1, min_support)
    item_set_list = [item_set_1]
    k = 2
    while len(item_set_list[k-2]) > 0:
        candidate_set_k = apriori_gen(item_set_list[k-2], k)
        # scan DB to get item_set_k_1
        item_set_k, support_data_k = scan_d(data_set, candidate_set_k, min_support)
        support_data.update(support_data_k)
        item_set_list.append(item_set_k)
        k += 1
    return item_set_list, support_data


def generate_rules(item_set_list, support_data, min_conf=0.7):
    """
    生成关联规则
    :param item_set_list: 频繁项集列表
    :param support_data: 包含频繁项集支持数据的字典
    :param min_conf: 最小可信度阈值
    :return: 包含可信度的规则列表
    """
    big_rule_list = []
    for i in range(1, len(item_set_list)):  # only get the sets with two or more items
        for freq_set in item_set_list[i]:
            h1 = [frozenset([item]) for item in freq_set]
            if i > 1:
                rules_from_conseq(freq_set, h1, support_data, big_rule_list, min_conf)
            else:
                calc_conf(freq_set, h1, support_data, big_rule_list, min_conf)
    return big_rule_list


def calc_conf(freq_set, each_item_list, support_data, big_rule_list, min_conf=0.7):
    """
    计算可信度值
    :param freq_set: 频繁项集
    :param each_item_list: 频繁项集里面只包含单个元素集合的列表
    :param support_data:
    :param big_rule_list:
    :param min_conf:
    :return:
    """
    pruned_h = []        # create new list to return
    for each_item_set in each_item_list:
        # calculate confidence
        conf = support_data[freq_set]/support_data[freq_set-each_item_set]
        if conf >= min_conf:
            print(freq_set - each_item_set, '-->', each_item_set, 'conf:', conf)
            big_rule_list.append((freq_set - each_item_set, each_item_set, conf))
            pruned_h.append(each_item_set)
    return pruned_h


def rules_from_conseq(freq_set, h, support_data, big_rule_list, min_conf=0.7):
    """
    生成候选规则集合
    :param freq_set:
    :param h:
    :param support_data:
    :param big_rule_list:
    :param min_conf:
    :return:
    """
    m = len(h[0])
    if len(freq_set) > (m + 1):         # try further merging
        hmp1 = apriori_gen(h, m + 1)    # create Hm+1 new candidates
        hmp1 = calc_conf(freq_set, hmp1, support_data, big_rule_list, min_conf)
        if len(hmp1) > 1:               # need at least two sets to merge
            rules_from_conseq(freq_set, hmp1, support_data, big_rule_list, min_conf)


def pnt_rules(rule_list, item_meaning):
    for ruleTup in rule_list:
        for item in ruleTup[0]:
            print(item_meaning[item])
        print("           -------->")
        for item in ruleTup[1]:
            print(item_meaning[item])
        print("confidence: %f" % ruleTup[2])
        print("\n")       # print a blank line
        

def get_action_ids():
    action_id_list = []
    bill_title_list = []
    fr = open('recent20bills.txt')
    vsmart = VoteSmartAPI(api_key="VOTE_SMART_API_KEY")
    for line in fr.readlines():
        bill_num = int(line.split('\t')[0])
        try:
            bill_detail = vsmart.Votes.getBill(bill_num)   # api call
            for action in bill_detail.actions:
                if action.level == 'House' and \
                (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    action_id = int(action.actionId)
                    print('bill: %d has actionId: %d' % (bill_num, action_id))
                    action_id_list.append(action_id)
                    bill_title_list.append(line.strip().split('\t')[1])
        except Exception as ex:
            print(ex)
            print("problem getting bill %d" % bill_num)
        sleep(1)                                      # delay to be polite
    return action_id_list, bill_title_list


def getTransList(actionIdList, billTitleList):
    """
    this will return a list of lists containing ints
    :param actionIdList:
    :param billTitleList:
    :return:
    """
    item_meaning = ['Republican', 'Democratic']     # list of what each item stands for
    for billTitle in billTitleList:             # fill up item_meaning list
        item_meaning.append('%s -- Nay' % billTitle)
        item_meaning.append('%s -- Yea' % billTitle)
    transDict = {}  # list of items in each transaction (politician)
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print('getting votes for actionId: %d' % actionId)
        try:
            voteList = VoteSmartAPI.Votes.getBillActionVotes(actionId)
            for vote in voteList:
                if vote.candidateName not in transDict:
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except Exception as ex:
            print(ex)
            print("problem getting actionId: %d" % actionId)
        voteCount += 2
    return transDict, item_meaning


def apriori_test():
    data_list = load_data_set()
    print("data list")
    print(data_list)
    # my_candidate_set_1 = create_c1(data_list)
    # print(my_candidate_set_1)
    # d = list(map(set, data_list))
    # print(d)
    # item_set_1, support_data0 = scan_d(d, my_candidate_set_1, 0.5)
    # print(item_set_1)
    freq_set_list, support_data = apriori(data_list)
    print("freq_set_list")
    print(freq_set_list)
    print("support data")
    print(support_data)
    print("generate rules")
    rules = generate_rules(freq_set_list, support_data)
    print("rules")
    print(rules)


def vote_test():
    action_ids,  bill_title_list = get_action_ids()
    print(action_ids)
    print(bill_title_list)


def mushroom_test():
    mush_data_set = [line.split() for line in open('mushroom.dat').readlines()]
    l, support_data = apriori(mush_data_set, min_support=0.3)
    for item in l[1]:
        if item.intersection('2'):
            print(item)
    for item in l[3]:
        if item.intersection('2'):
            print(item)


if __name__ == '__main__':
    # apriori_test()
    # vote_test()
    mushroom_test()
    print("Run apriori finish")
