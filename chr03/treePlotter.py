# coding=utf-8
"""
Created on Oct 14, 2010
@author: Peter Harrington
"""
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False    # 用来正常显示负号

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def get_num_leafs(_my_tree):
    num_leafs = 0
    first_str = list(_my_tree.keys())[0]
    second_dict = _my_tree[first_str]
    for key in second_dict.keys():
        # test to see if the nodes are dictionaries, if not they are leaf nodes
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


def get_tree_depth(_my_tree):
    max_depth = 0
    first_str = list(_my_tree.keys())[0]
    second_dict = _my_tree[first_str]
    for key in second_dict.keys():
        # test to see if the nodes are dictionaries, if not they are leaf nodes
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


def plot_node(node_text, center_pt, parent_pt, node_type):
    """
    参考：https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.annotate.html
    :param node_text:
    :param center_pt:
    :param parent_pt:
    :param node_type:
    :return:
    """
    create_plot.ax1.annotate(
        node_text, xy=parent_pt, xycoords='axes fraction',
        xytext=center_pt, textcoords='axes fraction',
        va="center", ha="center", bbox=node_type, arrowprops=arrow_args
    )


def plot_mid_text(cntr_pt, parent_pt, txt_string):
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid, txt_string, va="center", ha="center", rotation=30)


def plot_tree(_my_tree, parent_pt, _node_text):    # if the first key tells you what feat was split on
    num_leafs = get_num_leafs(_my_tree)      # this determines the x width of this tree
    depth = get_tree_depth(_my_tree)
    first_str = list(_my_tree.keys())[0]         # the text label for this node should be this
    cntr_pt = (plot_tree.xOff + (1.0 + float(num_leafs)) / 2.0 / plot_tree.totalW, plot_tree.yOff)
    plot_mid_text(cntr_pt, parent_pt, _node_text)
    plot_node(first_str, cntr_pt, parent_pt, decisionNode)
    second_dict = _my_tree[first_str]
    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.totalD
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            # test to see if the nodes are dictionaries, if not they are leaf nodes
            plot_tree(second_dict[key], cntr_pt, str(key))        # recursion
        else:
            # it's a leaf node print the leaf node
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalW
            plot_node(second_dict[key], (plot_tree.xOff, plot_tree.yOff), cntr_pt, leafNode)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), cntr_pt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.totalD


# def create_plot():
#     fig = plt.figure(1, facecolor='white')
#     fig.clf()
#     create_plot.ax1 = plt.subplot(111, frameon=False)
#     plot_node(U'决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
#     plot_node(U'叶子节点', (0.8, 0.1), (0.3, 0.8), leafNode)
#     plt.show()


def create_plot(in_tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    ax_props = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **ax_props)    # no ticks
    # createPlot.ax1 = plt.subplot(111, frameon=False) # ticks for demo puropses
    plot_tree.totalW = float(get_num_leafs(in_tree))
    plot_tree.totalD = float(get_tree_depth(in_tree))
    plot_tree.xOff = -0.5 / plot_tree.totalW
    plot_tree.yOff = 1.0
    plot_tree(in_tree, (0.5, 1.0), '')
    plt.show()


def retrieve_tree(i):
    list_of_trees = [
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    ]
    return list_of_trees[i]


if __name__ == '__main__':
    my_tree = retrieve_tree(0)
    # my_tree['no surfacing'][3] = "maybe"
    # print("my_tree")
    # print(my_tree)
    # print(get_num_leafs(my_tree))
    # print(get_tree_depth(my_tree))
    create_plot(my_tree)
    print("Run Decision Tree plot finish")