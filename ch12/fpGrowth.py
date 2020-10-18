"""
Created on Jun 14, 2011
FP-Growth FP means frequent pattern
the FP-Growth algorithm needs: 
1. FP-tree (class TreeNode)
2. header table (use dict)
This finds frequent item sets similar to apriori but does not find association rules.
@author: Peter
"""
import twitter
from time import sleep
import re


class TreeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode  # needs to be updated
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def display(self, ind=1):
        print('  ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.display(ind + 1)


def create_tree(dataSet, minSup=1):
    # create FP-tree from dataset but don't mine
    headerTable = {}
    # go over dataSet twice
    for trans in dataSet:  # first pass counts frequency of occurance
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    for k in list(headerTable.keys()):  # remove items not meeting minSup
        if headerTable[k] < minSup:
            del headerTable[k]
    freqItemSet = set(headerTable.keys())
    print('freqItemSet: ', freqItemSet)
    if len(freqItemSet) == 0:
        return None, None  # if no items meet min support -->get out
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]  # reformat headerTable to use Node link
    print('headerTable: ', headerTable)
    retTree = TreeNode('Null Set', 1, None)  # create tree
    for tranSet, count in dataSet.items():  # go through dataset 2nd time
        localD = {}
        for item in tranSet:  # put transaction items in order
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            # populate tree with ordered freq item set
            update_tree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable  # return tree and header table


def update_tree(items, inTree, headerTable, count):
    # check if orderedItems[0] in retTree.children
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)  # increment count
    else:  # add items[0] to inTree.children
        inTree.children[items[0]] = TreeNode(items[0], count, inTree)
        if headerTable[items[0]][1] is None:  # update header table
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            update_header(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:  # call updateTree() with remaining ordered items
        update_tree(items[1::], inTree.children[items[0]], headerTable, count)


def update_header(nodeToTest, targetNode):  # this version does not use recursion
    while nodeToTest.nodeLink is not None:  # Do not use recursion to traverse a linked list!
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def ascend_tree(leafNode, prefixPath):  # ascends from leaf node to root
    if leafNode.parent is not None:
        prefixPath.append(leafNode.name)
        ascend_tree(leafNode.parent, prefixPath)


def find_prefix_path(treeNode):  # TreeNode comes from header table
    condPats = {}
    while treeNode is not None:
        prefixPath = []
        ascend_tree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


def mine_tree(inTree, headerTable, minSup, preFix, freqItemList):
    # sort header table
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[0])]
    for basePat in bigL:  # start from bottom of header table
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        # append to set
        print('final frequent item: ', newFreqSet)
        freqItemList.append(newFreqSet)
        condPattBases = find_prefix_path(headerTable[basePat][1])
        print('condPattBases: ', basePat, condPattBases)
        # 2. construct cond FP-tree from cond. pattern base
        myCondTree, myHead = create_tree(condPattBases, minSup)
        print('head from conditional tree: ', myHead)
        if myHead is not None:  # 3. mine cond. FP-tree
            print('conditional tree for: ', newFreqSet)
            myCondTree.display(1)
            mine_tree(myCondTree, myHead, minSup, newFreqSet, freqItemList)


def load_simple_data():
    simple_data = [
        ['r', 'z', 'h', 'j', 'p'],
        ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
        ['z'],
        ['r', 'x', 'n', 'o', 's'],
        ['y', 'r', 'x', 'z', 'q', 't', 'p'],
        ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']
    ]
    return simple_data


def create_init_set(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


def text_parse(bigString):
    urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)
    listOfTokens = re.split(r'\W*', urlsRemoved)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def get_lots_of_tweets(searchStr):
    CONSUMER_KEY = ''
    CONSUMER_SECRET = ''
    ACCESS_TOKEN_KEY = ''
    ACCESS_TOKEN_SECRET = ''
    api = twitter.Api(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN_KEY,
                      access_token_secret=ACCESS_TOKEN_SECRET)
    # you can get 1500 results 15 pages * 100 per page
    resultsPages = []
    for i in range(1, 15):
        print("fetching page %d" % i)
        searchResults = api.GetSearch(searchStr, per_page=100, page=i)
        resultsPages.append(searchResults)
        sleep(6)
    return resultsPages


def mine_tweets(tweetArr, minSup=5):
    parsedList = []
    for i in range(14):
        for j in range(100):
            parsedList.append(text_parse(tweetArr[i][j].text))
    initSet = create_init_set(parsedList)
    my_fp_tree, myHeaderTab = create_tree(initSet, minSup)
    myFreqList = []
    mine_tree(my_fp_tree, myHeaderTab, minSup, set([]), myFreqList)
    return myFreqList


def hello_root_node():
    rootNode = TreeNode("pyramid", 9, None)
    rootNode.children["eye"] = TreeNode("eye", 13, None)
    rootNode.display()
    rootNode.children["phoenix"] = TreeNode("phoenix", 3, None)
    rootNode.display()


def hello_fp_tree():
    minSup = 3
    simpleData = load_simple_data()
    initSet = create_init_set(simpleData)
    myFPTree, myHeaderTab = create_tree(initSet, minSup)
    myFPTree.display()
    print(find_prefix_path(myHeaderTab['t'][1]))
    freqItems = []
    mine_tree(myFPTree, myHeaderTab, minSup, set([]), freqItems)
    print("freqItems:")
    print(freqItems)


def hello_kosarak():
    parseDat = [line.split() for line in open('kosarak.dat').readlines()]
    initSet = create_init_set(parseDat)
    my_fp_tree, myHeaderTab = create_tree(initSet, 100000)
    myFreqList = []
    mine_tree(my_fp_tree, myHeaderTab, 100000, set([]), myFreqList)
    print(len(myFreqList))
    print(myFreqList)


if __name__ == '__main__':
    # hello_root_node()
    # hello_fp_tree()
    hello_kosarak()
    print("Run fpGrowth finish")
