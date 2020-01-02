# encoding: UTF-8


def load_data_set(file_name):
    # general function to parse tab -delimited floats
    num_feat = len(open(file_name).readline().split('\t')) - 1  # get number of fields
    _data_mat = []
    _label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feat):
            line_arr.append(float(cur_line[i]))
        _data_mat.append(line_arr)
        _label_mat.append(float(cur_line[-1]))
    return _data_mat, _label_mat


if __name__ == '__main__':
    try:
        ex0_file = r"..\ch08\ex0.txt"
        data_mat, label_mat = load_data_set(ex0_file)
        print(data_mat)
        print(label_mat)
    except Exception as ex:
        print(ex)
    print("Run util finish")
