# coding=utf-8
"""
Created on Jan 8, 2011
@author: Peter
"""
from numpy import array, linalg, mat, shape, eye, zeros, ones, \
    inf, exp, mean, var, random, nonzero, multiply, corrcoef
from statsmodels import regression
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from common.util import load_data_set


def stand_regress(x_arr, y_arr):
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    x_tx = x_mat.T * x_mat
    if linalg.det(x_tx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    result = x_tx.I * (x_mat.T * y_mat)
    # 将matrix转化为ndarray
    return result.T.A[0]


def stand_regress_2(x_arr, y_arr):
    """
    调用statsmodels api进行线性回归
    :param x_arr:
    :param y_arr:
    :return: 回归参数，类型为ndarray
    """
    model = regression.linear_model.OLS(y_arr, x_arr).fit()
    return model.params


def lwlr(test_point, _x_arr, _y_arr, k=1.0):
    """
    局部加权线性回归
    :param test_point: x空间中的任意一点
    :param _x_arr:
    :param _y_arr:
    :param k:
    :return: 预测值，是一个数
    """
    x_mat = mat(_x_arr)
    y_mat = mat(_y_arr).T
    m = shape(x_mat)[0]
    weights = mat(eye(m))
    # next 2 lines create arr_weights matrix
    for j in range(m):
        diff_mat = test_point - x_mat[j, :]
        weights[j, j] = exp(diff_mat*diff_mat.T/(-2.0*k**2))
    x_tx = x_mat.T * (weights * x_mat)
    if linalg.det(x_tx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = x_tx.I * (x_mat.T * (weights * y_mat))
    return test_point * ws


def lwlr_array(test_arr, _x_arr, _y_arr, k=1.0):
    """
    局部加权线性回归测试
    :param test_arr:
    :param _x_arr:
    :param _y_arr:
    :param k:
    :return: 和test_arr相同大小的预测值矩阵
    """
    # loops over all the data points and applies lwlr to each one
    m = shape(test_arr)[0]
    _y_hat = zeros(m)
    for i in range(m):
        _y_hat[i] = lwlr(test_arr[i], _x_arr, _y_arr, k)
    return _y_hat


def lwlr_array_plot(x_arr, y_arr, k=1.0):
    """
    same thing as lwlr_array except it sorts X first, easier for plotting
    :param x_arr:
    :param y_arr:
    :param k:
    :return:
    """
    y_hat = zeros(shape(y_arr))
    x_copy = mat(x_arr)
    x_copy.sort(0)
    _size = shape(x_arr)[0]
    for i in range(_size):
        y_hat[i] = lwlr(x_copy[i], x_arr, y_arr, k)
    return y_hat, x_copy


def rss_error(y_arr, y_hat_arr):
    # y_arr and y_hat_arr both need to be arrays
    return ((y_arr - y_hat_arr) ** 2).sum()


def ridge_regression(x_mat, y_mat, lam=0.2):
    x_tx = x_mat.T * x_mat
    _size = shape(x_mat)[1]
    denom = x_tx + eye(_size) * lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (x_mat.T * y_mat)
    return ws


def ridge_test(x_arr, y_arr):
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    y_mean = mean(y_mat, 0)
    y_mat = y_mat - y_mean         # to eliminate X0 take mean off of Y
    # regularize X's
    x_means = mean(x_mat, 0)       # calc mean then subtract it off
    x_var = var(x_mat, 0)          # calc variance of Xi then divide by it
    x_mat = (x_mat - x_means)/x_var
    num_test_pts = 30
    num_feat_count = shape(x_mat)[1]
    w_mat = zeros((num_test_pts, num_feat_count))
    for i in range(num_test_pts):
        ws = ridge_regression(x_mat, y_mat, exp(i - 10))
        w_mat[i, :] = ws.T
    return w_mat


def regularize(x_mat):
    # regularize by columns
    in_mat = x_mat.copy()
    in_means = mean(in_mat, 0)          # calc mean then subtract it off
    in_var = var(in_mat, 0)             # calc variance of Xi then divide by it
    in_mat = (in_mat - in_means)/in_var
    return in_mat


def stage_wise(x_arr, y_arr, eps=0.01, num_it=100):
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    y_mean = mean(y_mat, 0)
    y_mat = y_mat - y_mean         # can also regularize ys but will get smaller coef
    x_mat = regularize(x_mat)
    m, n = shape(x_mat)
    return_mat = zeros((num_it, n))  # testing code remove
    ws = zeros((n, 1))
    ws_max = ws.copy()
    for i in range(num_it):
        # print(ws.T)
        lowest_error = inf  # numpy中用来表示正无穷大
        for j in range(n):
            for sign in [-1, 1]:
                ws_test = ws.copy()
                ws_test[j] += eps*sign
                y_test = x_mat * ws_test
                rss_e = rss_error(y_mat.A, y_test.A)
                if rss_e < lowest_error:
                    lowest_error = rss_e
                    ws_max = ws_test
        ws = ws_max.copy()
        return_mat[i, :] = ws.T
    return return_mat


def scrape_page(in_file, year, num_pieces, original_price):
    """
    通过解析本地html文件获取乐高玩具销售数据
    :param in_file: 本地html文件
    :param year:
    :param num_pieces:
    :param original_price:
    :return:
    """
    lego_x = []
    lego_y = []
    fr = open(in_file, 'r', encoding='utf-8')
    soup = BeautifulSoup(fr.read(), "lxml")
    i = 1
    current_row = soup.findAll('table', r="%d" % i)
    while len(current_row) != 0:
        title = current_row[0].findAll('a')[1].text
        lwr_title = title.lower()
        if (lwr_title.find('new') > -1) or (lwr_title.find('nisb') > -1):
            new_flag = 1.0
        else:
            new_flag = 0.0
        sold_span = current_row[0].findAll('td')[3].findAll('span')
        if len(sold_span) > 0:
            sold_price = current_row[0].findAll('td')[4]
            price_str = sold_price.text
            price_str = price_str.replace('$', '')     # strips out $
            price_str = price_str.replace(',', '')      # strips out ,
            if len(sold_price) > 1:
                price_str = price_str.replace('Free shipping', '')    # strips out Free Shipping
            lego_x.append([year, num_pieces, new_flag, original_price])
            lego_y.append(float(price_str))
        i += 1
        current_row = soup.findAll('table', r="%d" % i)
    fr.close()
    return lego_x, lego_y


def cross_validation(x_arr, y_arr, num_val=10):
    """
    交叉验证
    :param x_arr:
    :param y_arr:
    :param num_val: 交叉验证的次数
    :return:
    """
    m = len(y_arr)
    index_list = list(range(m))
    error_mat = zeros((num_val, 30))   # create error mat 30columns num_val rows
    for i in range(num_val):
        train_x, train_y = [], []
        test_x, test_y = [], []
        random.shuffle(index_list)
        # create training set based on first 90% of values in indexList
        for j in range(m):
            if j < m*0.9: 
                train_x.append(x_arr[index_list[j]])
                train_y.append(y_arr[index_list[j]])
            else:
                test_x.append(x_arr[index_list[j]])
                test_y.append(y_arr[index_list[j]])
        w_mat = ridge_test(train_x, train_y)        # get 30 weight vectors from ridge
        for k in range(30):                         # loop over all of the ridge estimates
            mat_test_x, mat_train_x = mat(test_x), mat(train_x)
            mean_train = mean(mat_train_x, 0)
            var_train = var(mat_train_x, 0)
            mat_test_x = (mat_test_x-mean_train)/var_train    # regularize test with training params
            y_est = mat_test_x * mat(w_mat[k, :]).T + mean(train_y)   # test ridge results and store
            error_mat[i, k] = rss_error(y_est.T.A, array(test_y))
            # print errorMat[i,k]
    mean_errors = mean(error_mat, 0)   # calc avg performance of the different ridge weight vectors
    min_mean = float(min(mean_errors))
    best_weights = w_mat[nonzero(mean_errors == min_mean)]
    # can unregularize to get model
    # when we regularized we wrote Xreg = (x-meanX)/var(x)
    # we can now write in terms of x not Xreg: x*w/var(x) - meanX/var(x) +meanY
    x_mat, y_mat = mat(x_arr), mat(y_arr).T
    mean_x, var_x = mean(x_mat, 0), var(x_mat, 0)
    un_reg = best_weights/var_x
    print("the best model from Ridge Regression is:", un_reg)
    print("with constant term: ", -1*sum(multiply(mean_x, un_reg)) + mean(y_mat))


def line_regression_test():
    try:
        x_arr, y_arr = load_data_set(r"ex0.txt")
        print(x_arr[0:2])
        ws = stand_regress(x_arr, y_arr)
        print("stand_regress result:{}".format(ws))
        ws2 = stand_regress_2(x_arr, y_arr)
        print("stand_regress_2 result:{}".format(ws2))
        x_array = array(x_arr)
        y_array = array(y_arr)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x_array[:, 1], y_array)
        y_hat = ws2[0] + x_array[:, 1] * ws2[1]
        print(corrcoef(y_hat, y_arr))
        ax.plot(x_array[:, 1], y_hat, 'r')
        plt.show()
    except Exception as ex:
        print(ex)


def lwlr_test():
    x_arr, y_arr = load_data_set(r"ex0.txt")
    # print(y_arr[0])
    # print(lwlr(x_arr[0], x_arr, y_arr, 1.0))
    # print(lwlr(x_arr[0], x_arr, y_arr, 0.001))
    fig = plt.figure()
    k_list = [1, 0.1, 0.003]
    for index in range(3):
        y_hat, x_sort = lwlr_array_plot(x_arr, y_arr, k_list[index])
        ax = fig.add_subplot(3, 1, index+1)
        ax.plot(x_sort[:, 1], y_hat)
        ax.scatter(mat(x_arr)[:, 1].flatten().A[0], mat(y_arr).T.flatten().A[0], s=2, c='red')
    plt.show()


def abalone_test():
    """
    预测鲍鱼的年龄
    :return:
    """
    ab_x, ab_y = load_data_set(r"abalone.txt")
    print("ab_x shape:{}, ab_y shape:{}".format(shape(ab_x), shape(ab_y)))
    y_hat_01 = lwlr_array(ab_x[0:99], ab_x[0:99], ab_y[0:99], 0.1)
    y_hat_1 = lwlr_array(ab_x[0:99], ab_x[0:99], ab_y[0:99], 1)
    y_hat_10 = lwlr_array(ab_x[0:99], ab_x[0:99], ab_y[0:99], 10)
    print("训练集，k=0.1, rss_error:{}".format(rss_error(ab_y[0:99], y_hat_01.T)))
    print("训练集，k=1, rss_error:{}".format(rss_error(ab_y[0:99], y_hat_1.T)))
    print("训练集，k=10, rss_error:{}".format(rss_error(ab_y[0:99], y_hat_10.T)))
    # 检验训练集误差
    y_hat_01 = lwlr_array(ab_x[100:199], ab_x[0:99], ab_y[0:99], 0.1)
    print("测试集，k=0.1, rss_error:{}".format(rss_error(ab_y[100:199], y_hat_01.T)))
    y_hat_1 = lwlr_array(ab_x[100:199], ab_x[0:99], ab_y[0:99], 1)
    print("测试集，k=1, rss_error:{}".format(rss_error(ab_y[100:199], y_hat_1.T)))
    y_hat_10 = lwlr_array(ab_x[100:199], ab_x[0:99], ab_y[0:99], 10)
    print("测试集，k=10 rss_error:{}".format(rss_error(ab_y[100:199], y_hat_10.T)))
    ws = stand_regress(ab_x[0:99], ab_y[0:99])
    y_hat = mat(ab_x)[100:199] * mat(ws).T
    print("测试集，简单线性回归 rss_error:{}".format(rss_error(ab_y[100:199], y_hat.T.A[0])))


def ridge_plot_test():
    ab_x, ab_y = load_data_set(r"abalone.txt")
    ridge_weights = ridge_test(ab_x, ab_y)
    print(ridge_weights)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridge_weights)
    plt.show()


def stage_test():
    ab_x, ab_y = load_data_set(r"abalone.txt")
    weights = stage_wise(ab_x, ab_y, 0.01, 200)
    print("前向线性回归，eps:{}, num_it:{}".format(0.01, 200))
    print(weights[-1])
    weights = stage_wise(ab_x, ab_y, 0.001, 5000)
    print("前向线性回归，eps:{}, num_it:{}".format(0.001, 5000))
    print(weights[-1])
    x_mat = mat(ab_x)
    y_mat = mat(ab_y).T
    x_mat = regularize(x_mat)
    y_mean = mean(y_mat, 0)
    y_mat = y_mat - y_mean
    weights = stand_regress(x_mat, y_mat.T)
    print("标准线性回归")
    print(weights)


def lego_test():
    lg_x = []
    lg_y = []
    template = r"setHtml/lego{}.html"
    print_template = "scrape page lego{}.html finish, size:{}"
    data = (
        ("8288", 2006, 800, 49.99),
        ("10030", 2002, 3096, 269.99),
        ("10179", 2007, 5195, 499.99),
        ("10181", 2007, 3428, 199.99),
        ("10189", 2008, 5922, 299.99),
        ("10196", 2009, 3263, 249.99)
    )
    for index in range(len(data)):
        result = scrape_page(
            template.format(data[index][0]), data[index][1], data[index][2], data[index][3])
        print(print_template.format(data[index][0], len(result[0])))
        lg_x.extend(result[0])
        lg_y.extend(result[1])
    print("lg_x shape:{}".format(shape(lg_x)))
    print("lg_y shape:{}".format(shape(lg_y)))
    size = shape(lg_y)[0]
    lg_x1 = mat(ones((size, 5)))
    print("lg_x[0]:{}".format(lg_x[0]))
    lg_x1[:, 1:5] = mat(lg_x)
    print("lg_x1[0]:{}".format(lg_x1[0]))
    ws = mat(stand_regress(lg_x1, mat(lg_y)))
    print("ws:{}".format(ws.A[0]))
    print("lg_x1[0] * ws:{}, lg_y[0]:{}".format(round((lg_x1[0]*ws.T).A[0][0], 2), lg_y[0]))
    print("lg_x1[-1] * ws:{}, lg_y[0]:{}".format(round((lg_x1[-1]*ws.T).A[0][0], 2), lg_y[-1]))
    print("lg_x1[43] * ws:{}, lg_y[0]:{}".format(round((lg_x1[43]*ws.T).A[0][0], 2), lg_y[43]))
    cross_validation(lg_x, lg_y, 10)


if __name__ == '__main__':
    # line_regression_test()
    # lwlr_test()
    # ridge_plot_test()
    # stage_test()
    lego_test()
    print("Run regression finish")
