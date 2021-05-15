# coding:utf-8
"""
main module of mondrian
"""


import pdb
import time
from utility import cmp_value, value, merge_qi_value
from functools import cmp_to_key
from read_adult_data import read_data as read_adult
import sys, copy, random
# warning all these variables should be re-inited, if
# you want to run mondrian with different parameters
DATA_SELECT = 'a'
RELAX = False
INTUITIVE_ORDER = None
__DEBUG = False
QI_LEN = 2
GL_K = 0
RESULT = []
QI_RANGE = []
QI_DICT = []
QI_ORDER = []

class Partition(object):

    """
    Class for Group (or EC), which is used to keep records
    self.member: records in group
    self.low: lower point, use index to avoid negative values
    self.high: higher point, use index to avoid negative values
    self.allow: show if partition can be split on this QI
    """

    def __init__(self, data, low, high):
        """
        split_tuple = (index, low, high)
        """
        self.low = list(low)
        self.high = list(high)
        self.member = data[:]
        self.allow = [1] * QI_LEN

    def add_record(self, record, dim):
        """
        add one record to member
        """
        self.member.append(record)

    def add_multiple_record(self, records, dim):
        """
        add multiple records (list) to partition
        """
        for record in records:
            self.add_record(record, dim)

    def __len__(self):
        """
        return number of records
        """
        return len(self.member)


def get_normalized_width(partition, index):
    """
    return Normalized width of partition
    similar to NCP
    """
    d_order = QI_ORDER[index]
    width = value(d_order[partition.high[index]]) - value(d_order[partition.low[index]])
    if width == QI_RANGE[index]:
        return 1
    return width * 1.0 / QI_RANGE[index]


def choose_dimension(partition):
    """
    choose dim with largest norm_width from all attributes.
    This function can be upgraded with other distance function.
    """
    max_width = -1
    max_dim = -1
    for dim in range(QI_LEN):
        if partition.allow[dim] == 0:
            continue
        norm_width = get_normalized_width(partition, dim)
        if norm_width > max_width:
            max_width = norm_width
            max_dim = dim
    if max_width > 1:
        pdb.set_trace()
    return max_dim


def frequency_set(partition, dim):
    """
    get the frequency_set of partition on dim
    """
    frequency = {}
    for record in partition.member:
        try:
            frequency[record[dim]] += 1
        except KeyError:
            frequency[record[dim]] = 1
    return frequency


def find_median(partition, dim):
    """
    find the middle of the partition, return split_val
    """
    # use frequency set to get median
    frequency = frequency_set(partition, dim)
    split_val = ''
    next_val = ''
    value_list = list(frequency.keys())
    value_list.sort(key=cmp_to_key(cmp_value))
    total = sum(frequency.values())
    middle = total // 2
    if middle < GL_K or len(value_list) <= 1:
        try:
            return '', '', value_list[0], value_list[-1]
        except IndexError:
            return '', '', '', ''
    index = 0
    split_index = 0
    for i, qi_value in enumerate(value_list):
        index += frequency[qi_value]
        if index >= middle:
            split_val = qi_value
            split_index = i
            break
    else:
        print("Error: cannot find split_val")
    try:
        next_val = value_list[split_index + 1]
    except IndexError:
        # there is a frequency value in partition
        # which can be handle by mid_set
        # e.g.[1, 2, 3, 4, 4, 4, 4]
        next_val = split_val
    return (split_val, next_val, value_list[0], value_list[-1])


def anonymize_strict(partition):
    """
    recursively partition groups until not allowable
    """
    allow_count = sum(partition.allow)
    # only run allow_count times
    if allow_count == 0:
        RESULT.append(partition)
        return
    for index in range(allow_count):
        # choose attrubite from domain
        dim = choose_dimension(partition)
        if dim == -1:
            print("Error: dim=-1")
            pdb.set_trace()
        (split_val, next_val, low, high) = find_median(partition, dim)
        # Update parent low and high
        if low != '':
            partition.low[dim] = QI_DICT[dim][low]
            partition.high[dim] = QI_DICT[dim][high]
        if split_val == '' or split_val == next_val:
            # cannot split
            partition.allow[dim] = 0
            continue
        # split the group from median
        mean = QI_DICT[dim][split_val]
        lhs_high = partition.high[:]
        rhs_low = partition.low[:]
        lhs_high[dim] = mean
        rhs_low[dim] = QI_DICT[dim][next_val]
        lhs = Partition([], partition.low, lhs_high)
        rhs = Partition([], rhs_low, partition.high)
        for record in partition.member:
            pos = QI_DICT[dim][record[dim]]
            if pos <= mean:
                # lhs = [low, mean]
                lhs.add_record(record, dim)
            else:
                # rhs = (mean, high]
                rhs.add_record(record, dim)
        # check is lhs and rhs satisfy k-anonymity
        if len(lhs) < GL_K or len(rhs) < GL_K:
            partition.allow[dim] = 0
            continue
        # anonymize sub-partition
        anonymize_strict(lhs)
        anonymize_strict(rhs)
        return
    RESULT.append(partition)



def init(data, k, QI_num=-1):
    """
    reset global variables
    """
    global GL_K, RESULT, QI_LEN, QI_DICT, QI_RANGE, QI_ORDER
    if QI_num <= 0:
        QI_LEN = len(data[0]) - 1
    else:
        QI_LEN = QI_num
    GL_K = k
    RESULT = []
    # static values
    QI_DICT = []
    QI_ORDER = []
    QI_RANGE = []
    att_values = []
    for i in range(QI_LEN):
        att_values.append(set())
        QI_DICT.append(dict())
    for record in data:
        for i in range(QI_LEN):
            att_values[i].add(record[i])
    for i in range(QI_LEN):
        value_list = list(att_values[i])
        value_list.sort(key=cmp_to_key(cmp_value))
        QI_RANGE.append(value(value_list[-1]) - value(value_list[0]))
        QI_ORDER.append(list(value_list))
        for index, qi_value in enumerate(value_list):
            QI_DICT[i][qi_value] = index


def mondrian(data, k, relax=False, QI_num=-1):
    """
    Main function of mondrian, return result in tuple (result, (ncp, rtime)).
    data: dataset in 2-dimensional array.
    k: k parameter for k-anonymity
    QI_num: Default -1, which exclude the last column. Othewise, [0, 1,..., QI_num - 1]
            will be anonymized, [QI_num,...] will be excluded.
    relax: determine use strict or relaxed mondrian,
    Both mondrians split partition with binary split.
    In strict mondrian, lhs and rhs have not intersection.
    But in relaxed mondrian, lhs may be have intersection with rhs.
    """
    init(data, k, QI_num)
    result = []
    data_size = len(data)
    low = [0] * QI_LEN
    high = [(len(t) - 1) for t in QI_ORDER]
    whole_partition = Partition(data, low, high)
    # begin mondrian
    start_time = time.time()

    # strict model
    anonymize_strict(whole_partition)
    rtime = float(time.time() - start_time)
    # generalization result and
    # evaluation information loss
    ncp = 0.0
    dp = 0.0
    for partition in RESULT:
        rncp = 0.0
        for index in range(QI_LEN):
            rncp += get_normalized_width(partition, index)
        rncp *= len(partition)
        ncp += rncp
        dp += len(partition) ** 2
        for record in partition.member[:]:
            for index in range(QI_LEN):
                record[index] = merge_qi_value(QI_ORDER[index][partition.low[index]],
                                QI_ORDER[index][partition.high[index]])
            result.append(record)
    # If you want to get NCP values instead of percentage
    # please remove next three lines
    ncp /= QI_LEN
    ncp /= data_size
    ncp *= 100
    if __DEBUG:
        from decimal import Decimal
        print("Discernability Penalty=%.2E" % Decimal(str(dp)))
        print("size of partitions=%d" % len(RESULT))
        print("K=%d" % k)
        print("NCP = %.2f %%" % ncp)
    return (result, (ncp, rtime))



def write_to_file(result):
    """
    write the anonymized result to anonymized.data
    """
    with open("anonymized.data", "w") as output:
        for r in result:
            output.write(';'.join(r) + '\n')


def get_result_one(data, k=10):
    """
    run mondrian for one time, with k=10
    """
    print("K=%d" % k)
    data_back = copy.deepcopy(data)
    result, eval_result = mondrian(data, k, RELAX)
    # Convert numerical values back to categorical values if necessary
    if DATA_SELECT == 'a':
        result = covert_to_raw(result)
    else:
        for r in result:
            r[-1] = ','.join(r[-1])
    # write to anonymized.out
    write_to_file(result)
    data = copy.deepcopy(data_back)
    print("NCP %0.2f" % eval_result[0] + "%")
    print("Running time %0.2f" % eval_result[1] + " seconds")


def get_result_k(data):
    """
    change k, while fixing QD and size of data set
    """
    data_back = copy.deepcopy(data)
    for k in range(5, 105, 5):
        print('#' * 30)
        print("K=%d" % k)
        result, eval_result = mondrian(data, k, RELAX)
        if DATA_SELECT == 'a':
            result = covert_to_raw(result)
        data = copy.deepcopy(data_back)
        print("NCP %0.2f" % eval_result[0] + "%")
        print("Running time %0.2f" % eval_result[1] + " seconds")


def get_result_dataset(data, k=10, num_test=10):
    """
    fix k and QI, while changing size of data set
    num_test is the test number.
    """
    data_back = copy.deepcopy(data)
    length = len(data_back)
    joint = 5000
    datasets = []
    check_time = length / joint
    if length % joint == 0:
        check_time -= 1
    for i in range(check_time):
        datasets.append(joint * (i + 1))
    datasets.append(length)
    ncp = 0
    rtime = 0
    for pos in datasets:
        print('#' * 30)
        print("size of dataset %d" % pos)
        for j in range(num_test):
            temp = random.sample(data, pos)
            result, eval_result = mondrian(temp, k, RELAX)
            if DATA_SELECT == 'a':
                result = covert_to_raw(result)
            ncp += eval_result[0]
            rtime += eval_result[1]
            data = copy.deepcopy(data_back)
        ncp /= num_test
        rtime /= num_test
        print("Average NCP %0.2f" % ncp + "%")
        print("Running time %0.2f" % rtime + " seconds")
        print('#' * 30)


def get_result_qi(data, k=10):
    """
    change number of QI, while fixing k and size of data set
    """
    data_back = copy.deepcopy(data)
    num_data = len(data[0])
    for i in reversed(list(range(1, num_data))):
        print('#' * 30)
        print("Number of QI=%d" % i)
        result, eval_result = mondrian(data, k, RELAX, i)
        if DATA_SELECT == 'a':
            result = covert_to_raw(result)
        data = copy.deepcopy(data_back)
        print("NCP %0.2f" % eval_result[0] + "%")
        print("Running time %0.2f" % eval_result[1] + " seconds")


def covert_to_raw(result, connect_str='~'):
    """
    During preprocessing, categorical attributes are covert to
    numeric attribute using intuitive order. This function will covert
    these values back to they raw values. For example, Female and Male
    may be converted to 0 and 1 during anonymizaiton. Then we need to transform
    them back to original values after anonymization.
    """
    covert_result = []
    qi_len = len(INTUITIVE_ORDER)
    for record in result:
        covert_record = []
        for i in range(qi_len):
            if len(INTUITIVE_ORDER[i]) > 0:
                vtemp = ''
                if connect_str in record[i]:
                    temp = record[i].split(connect_str)
                    raw_list = []
                    for j in range(int(temp[0]), int(temp[1]) + 1):
                        raw_list.append(INTUITIVE_ORDER[i][j])
                    vtemp = connect_str.join(raw_list)
                else:
                    vtemp = INTUITIVE_ORDER[i][int(record[i])]
                covert_record.append(vtemp)
            else:
                covert_record.append(record[i])
        if isinstance(record[-1], str):
            covert_result.append(covert_record + [record[-1]])
        else:
            covert_result.append(covert_record + [connect_str.join(record[-1])])
    return covert_result


if __name__ == '__main__':
    FLAG = ''
    try:
        MODEL = sys.argv[1]
        DATA_SELECT = sys.argv[2]
    except IndexError:
        MODEL = 's'
        DATA_SELECT = 'a'
    INPUT_K = 10
    # read record

    RELAX = False
    print("Strict Mondrian")
    print("Adult data")
        # INTUITIVE_ORDER is an intuitive order for
        # categorical attributes. This order is produced
        # by the reading (from data set) order.
    DATA, INTUITIVE_ORDER = read_adult()
    print(INTUITIVE_ORDER)


    get_result_one(DATA, INPUT_K)
    # anonymized dataset is stored in result
    print("Finish Mondrian!!")