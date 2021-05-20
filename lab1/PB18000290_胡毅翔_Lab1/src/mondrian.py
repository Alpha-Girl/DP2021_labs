import time
from utility import cmp_value, value, merge_qi_value, covert_to_raw
from functools import cmp_to_key
import copy
"""
K-anonymity
"""
K = 10

"""
here reset following var to use mondrian on other attributes
"""
INTUITIVE_ORDER = None
QI_LEN = 5
GLOBAL_K = 0
RESULT = []
QI_RANGE = []
QI_DICT = []
QI_ORDER = []

# Read data
ATTRIBUTE_NAME = ['age', 'work_class', 'final_weight', 'education',
                  'education_num', 'marital_status', 'occupation', 'relationship',
                  'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
                  'native_country', 'class']
# QI attributes
#          ['age', 'education_num', 'marital_status',
#            'race', 'sex','native_country']
QI_INDEX = [0, 4, 5, 8, 9]
# True: categorical , False: numeric
IS_CATEGORICAL = [False, False, True, True, True]
# sensitive attribute
SA_INDEX = 6


def read_data():
    QI_num = len(QI_INDEX)
    data = []
    # order categorical attributes in intuitive order
    # here, we use the appear number
    intuitive_dictionary = []
    intuitive_order = []
    intuitive_number = []
    # init intuitive
    for i in range(QI_num):
        intuitive_dictionary.append(dict())
        intuitive_number.append(0)
        intuitive_order.append(list())
    # open data
    data_file = open('adult.data', 'r')
    # read line by line
    for line in data_file:
        line = line.strip()
        # remove empty and incomplete lines
        # only 30162 records will be kept
        if len(line) == 0 or '?' in line:
            continue
        # remove double spaces
        line = line.replace(' ', '')
        temp = line.split(',')
        lst_temp = []
        for i in range(QI_num):
            index = QI_INDEX[i]
            # transform categorical to numeric
            if IS_CATEGORICAL[i]:
                try:
                    lst_temp.append(intuitive_dictionary[i][temp[index]])
                except KeyError:
                    intuitive_dictionary[i][temp[index]] = intuitive_number[i]
                    lst_temp.append(intuitive_number[i])
                    intuitive_number[i] += 1
                    intuitive_order[i].append(temp[index])
            else:
                lst_temp.append(int(temp[index]))
        lst_temp.append(temp[SA_INDEX])
        data.append(lst_temp)
    return data, intuitive_order


class Partition(object):
    """
    Class for Group (or EC), which is used to keep records
    self.member: records in group
    self.low: lower point
    self.high: higher point
    self.allowable: show if partition can be split on this QI
    """

    def __init__(self, data, low, high):
        self.low = list(low)
        self.high = list(high)
        self.member = data[:]
        self.allowable = [1] * QI_LEN

    def add_record(self, record, dim):
        self.member.append(record)

    def add_multiple_record(self, records, dim):
        for record in records:
            self.add_record(record, dim)

    def __len__(self):
        return len(self.member)


def get_LM(partition, index):
    """
    return LM
    """
    d_order = QI_ORDER[index]
    width = value(d_order[partition.high[index]]) - \
        value(d_order[partition.low[index]])
    if width == QI_RANGE[index]:
        return 1
    return width * 1.0 / QI_RANGE[index]


def choose_dimension(partition):
    """
    choose dim with largest LM from all attributes.
    """
    max_width = -1
    max_dim = -1
    for dim in range(QI_LEN):
        if partition.allowable[dim] == 0:
            continue
        norm_width = get_LM(partition, dim)
        if norm_width > max_width:
            max_width = norm_width
            max_dim = dim
    return max_dim


def frequency_set(partition, dim):
    """
    get the frequency_set of partition on dimension
    """
    frequency = {}
    for record in partition.member:
        try:
            frequency[record[dim]] += 1
        except KeyError:
            frequency[record[dim]] = 1
    return frequency


def find_median(frequency):
    """
    use frequency set to get median
    """
    split_Val = ''
    next_Val = ''
    value_list = list(frequency.keys())
    value_list.sort(key=cmp_to_key(cmp_value))
    # select median
    total = sum(frequency.values())
    middle = total // 2
    # judge if median satisfy k
    if middle < GLOBAL_K or len(value_list) <= 1:
        try:
            return '', '', value_list[0], value_list[-1]
        except IndexError:
            return '', '', '', ''
    index = 0
    split_index = 0
    # find out the split_Val
    for i, qi_value in enumerate(value_list):
        index += frequency[qi_value]
        if index >= middle:
            split_Val = qi_value
            split_index = i
            break
    try:
        next_Val = value_list[split_index + 1]
    except IndexError:
        next_Val = split_Val
    return (split_Val, next_Val, value_list[0], value_list[-1])


def anonymize(partition):
    """
    recursively partition groups until not allowable
    """
    count_allow = sum(partition.allowable)
    if count_allow == 0:
        RESULT.append(partition)
        return
    for index in range(count_allow):
        dim = choose_dimension(partition)
        frequency = frequency_set(partition, dim)
        (split_Val, next_Val, low, high) = find_median(frequency)
        # Update parent low and high
        if low != '':
            partition.low[dim] = QI_DICT[dim][low]
            partition.high[dim] = QI_DICT[dim][high]
        if split_Val == '' or split_Val == next_Val:
            # cannot split
            partition.allowable[dim] = 0
            continue
        # split the group from median
        mean = QI_DICT[dim][split_Val]
        lhs_high = partition.high[:]
        rhs_low = partition.low[:]
        lhs_high[dim] = mean
        rhs_low[dim] = QI_DICT[dim][next_Val]
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
        if len(lhs) < GLOBAL_K or len(rhs) < GLOBAL_K:
            partition.allowable[dim] = 0
            continue
        # anonymize sub-partition
        anonymize(lhs)
        anonymize(rhs)
        return
    RESULT.append(partition)


def init(data, k, QI_num=-1):
    """
    reset global variables
    """
    global GLOBAL_K, RESULT, QI_LEN, QI_DICT, QI_RANGE, QI_ORDER
    if QI_num <= 0:
        QI_LEN = len(data[0]) - 1
    else:
        QI_LEN = QI_num
    GLOBAL_K = k
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


def mondrian(data, k, QI_num=-1):
    """
    Main function of mondrian, return result in tuple (result, (LM, run_time)).
    data: dataset in 2-dimensional array.
    k: k parameter for k-anonymity
    QI_num: Default -1, which exclude the last column. Othewise, [0, 1,..., QI_num - 1]
            will be anonymized, [QI_num,...] will be excluded.
    relax: determine use strict or relaxed mondrian,
    Both mondrians split partition with binary split.
    """
    init(data, k, QI_num)
    result = []
    data_size = len(data)
    # init Partition
    low = [0] * QI_LEN
    high = [(len(t) - 1) for t in QI_ORDER]
    whole_partition = Partition(data, low, high)
    # begin mondrian
    start_time = time.time()
    anonymize(whole_partition)
    run_time = float(time.time() - start_time)
    # generalization result and evaluation information loss
    LM = 0.0
    for partition in RESULT:
        rLM = 0.0
        for index in range(QI_LEN):
            rLM += get_LM(partition, index)
        rLM *= len(partition)
        LM += rLM
        for record in partition.member[:]:
            for index in range(QI_LEN):
                record[index] = merge_qi_value(QI_ORDER[index][partition.low[index]],
                                               QI_ORDER[index][partition.high[index]])
            result.append(record)
    LM /= data_size
    return (result, (LM, run_time))


def write_result_data(result):
    """
    write the anonymized result to adult_mondrian.data
    """
    with open("adult_mondrian.data", "w") as output:
        for r in result:
            output.write(','.join(r) + '\n')


def run_Mondrian(data, k=10):
    print("K=%d" % k)
    data_back = copy.deepcopy(data)
    result, eval_result = mondrian(data, k)
    # Convert numerical values back to categorical values if necessary
    result = covert_to_raw(result, INTUITIVE_ORDER)
    # write to adult_mondrian
    write_result_data(result)
    data = copy.deepcopy(data_back)
    # print LM and time
    print("LM %.2f" % eval_result[0])
    print("Running time %.2f" % eval_result[1] + " seconds")


if __name__ == '__main__':
    # read data
    INPUT_DATA, INTUITIVE_ORDER = read_data()
    run_Mondrian(INPUT_DATA, K)
    print("Finish Mondrian!!")
