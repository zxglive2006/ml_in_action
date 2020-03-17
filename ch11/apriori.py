# encoding: UTF-8
"""
Created on Mar 24, 2011
Ch 11 code
@author: Peter
参考：
https://en.wikipedia.org/wiki/Apriori_algorithm
https://towardsdatascience.com/underrated-machine-learning-algorithms-apriori-1b1d7a8b7bc
"""
from numpy import *


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
    :return: 包含支持度值的列表
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


def apriori_gen(lk, k):
    # creates c_k
    ret_list = []
    len_lk = len(lk)
    for i in range(len_lk):
        for j in range(i+1, len_lk): 
            lst_1 = list(lk[i])[:k-2]
            lst_2 = list(lk[j])[:k-2]
            lst_1.sort()
            lst_2.sort()
            if lst_1 == lst_2:  # if first k-2 elements are equal
                ret_list.append(lk[i] | lk[j])   # set union
    return ret_list


def apriori(data_list, min_support=0.5):
    """

    :param data_list:
    :param min_support:
    :return:
    """
    candidate_set_1 = create_c1(data_list)
    data_set = list(map(set, data_list))
    item_set_1, support_data = scan_d(data_set, candidate_set_1, min_support)
    item_set_list = [item_set_1]
    k = 2
    while len(item_set_list[k-2]) > 0:
        candidate_set_k = apriori_gen(item_set_list[k - 2], k)
        # scan DB to get lk
        Lk, supK = scan_d(data_set, candidate_set_k, min_support)
        support_data.update(supK)
        item_set_list.append(Lk)
        k += 1
    return item_set_list, support_data


def generateRules(L, supportData, minConf=0.7):
    # supportData is a dict coming from scanD
    bigRuleList = []
    for i in range(1, len(L)):  # only get the sets with two or more items
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         


def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] #create new list to return
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence
        if conf >= minConf: 
            print(freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if len(freqSet) > (m + 1):      # try further merging
        Hmp1 = apriori_gen(H, m + 1)   # create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if len(Hmp1) > 1:           # need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print(itemMeaning[item])
        print("           -------->")
        for item in ruleTup[1]:
            print(itemMeaning[item])
        print("confidence: %f" % ruleTup[2])
        print("\n")       # print a blank line
        
            
from time import sleep
# from votesmart import votesmart
# votesmart.apikey = 'get your api key first'
# def getActionIds():
# #     actionIdList = []; billTitleList = []
# #     fr = open('recent20bills.txt')
# #     for line in fr.readlines():
# #         billNum = int(line.split('\t')[0])
# #         try:
# #             billDetail = votesmart.votes.getBill(billNum) #api call
# #             for action in billDetail.actions:
# #                 if action.level == 'House' and \
# #                 (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
# #                     actionId = int(action.actionId)
# #                     print('bill: %d has actionId: %d' % (billNum, actionId))
# #                     actionIdList.append(actionId)
# #                     billTitleList.append(line.strip().split('\t')[1])
# #         except:
# #             print("problem getting bill %d" % billNum)
# #         sleep(1)                                      #delay to be polite
# #     return actionIdList, billTitleList
        
# def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
#     itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
#     for billTitle in billTitleList:#fill up itemMeaning list
#         itemMeaning.append('%s -- Nay' % billTitle)
#         itemMeaning.append('%s -- Yea' % billTitle)
#     transDict = {}#list of items in each transaction (politician)
#     voteCount = 2
#     for actionId in actionIdList:
#         sleep(3)
#         print('getting votes for actionId: %d' % actionId)
#         try:
#             voteList = votesmart.votes.getBillActionVotes(actionId)
#             for vote in voteList:
#                 if not vote.candidateName in transDict:
#                     transDict[vote.candidateName] = []
#                     if vote.officeParties == 'Democratic':
#                         transDict[vote.candidateName].append(1)
#                     elif vote.officeParties == 'Republican':
#                         transDict[vote.candidateName].append(0)
#                 if vote.action == 'Nay':
#                     transDict[vote.candidateName].append(voteCount)
#                 elif vote.action == 'Yea':
#                     transDict[vote.candidateName].append(voteCount + 1)
#         except:
#             print("problem getting actionId: %d" % actionId)
#         voteCount += 2
#     return transDict, itemMeaning


def apriori_test():
    my_data_set = load_data_set()
    print(my_data_set)
    # my_c1 = create_c1(my_data_set)
    # print(my_c1)
    # d = list(map(set, my_data_set))
    # print(d)
    # l1, support_data0 = scan_d(d, my_c1, 0.5)
    # print(l1)
    l, support_data = apriori(my_data_set, min_support=0.7)
    print(l)
    # print(apriori_gen(l[0], 2))


if __name__ == '__main__':
    apriori_test()
    print("Run apriori finish")
