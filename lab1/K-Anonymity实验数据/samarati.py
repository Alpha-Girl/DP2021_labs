from numpy.lib.function_base import average
import pandas as pd
import time
import csv

# read data
raw_data = []
with open('test.csv')as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        raw_data.append(row)

# read Generalization tree
gender_tree = pd.read_csv(r'adult_gender.txt')
gender_tree = gender_tree.values.tolist()
race_tree = pd.read_csv(r'adult_race.txt')
race_tree = race_tree.values.tolist()
marital_status_tree = pd.read_csv(r'adult_marital_status.txt')
marital_status_tree = marital_status_tree.values.tolist()

# Generalization for Categorical attribute


def Generalization(tree, x, n):
    l = len(tree)
    if n == 0:
        return x
    else:
        for k in range(l):
            if tree[k][0] == x:
                return Generalization(tree, tree[k][1], n-1)

# Generalization for Numerical attribute


def age_Generalization(x, n):
    if n == 4:
        return "*"
    elif n == 0:
        return x
    else:
        m = 5*(2**(n-1))
        tmp = (int)(eval(x) / m)
        return str(tmp*m)+"-"+str(tmp*m+m)


'''
function:satisfy
input:  k (k-Anonymity),
        vec (generalization level for all attribute)
        MaxSup (max number of deleted records )
output: g (data after k-Anonymity),
        IsSatisfied (bool, True/Flase)
        LM (Loss Metric)
'''


def satisfy(k, vec, MaxSup):
    # t: data that before generalize
    t = raw_data
    # g: to store data that after generalize
    s_data = []
    with open('test.csv')as ss:
        ss_csv = csv.reader(ss)
        for row in ss_csv:
            s_data.append(row)
    g = []
    g = s_data
    l = len(t)
    # dictionary to count the same record
    d = {}
    # generalize
    for i in range(l):
        g[i][0] = age_Generalization(t[i][0], vec[0])
        g[i][1] = Generalization(gender_tree, t[i][1], vec[1])
        g[i][2] = Generalization(race_tree, t[i][2], vec[2])
        g[i][3] = Generalization(marital_status_tree, t[i][3], vec[3])
        # add to dictionary
        x = str(g[i][0])+str(g[i][1])+str(g[i][2])+str(g[i][3])
        tmp = d.get(x)
        if tmp == None:
            d[x] = 1
        else:
            d[x] = tmp+1
    # satisfy MaxSup or not
    count = 0
    tmp = list(d.keys())
    for j in range(len(tmp)):
        if d.get(tmp[j]) < k:
            count = count+d.get(tmp[j])
        if count > MaxSup:
            return [], False, float("inf")
    # delete records that less than k
    for j in range(len(tmp)):
        if d.get(tmp[j]) < k:
            delete = tmp[j]
            for i in g:
                x = str(i[0])+str(i[1])+str(i[2])+str(i[3])
                if x == delete:
                    g.remove(i)
    # calculate lm for marital_status_tree
    sum = 0
    if vec[3] == 1:
        for i in g:
            if i[3] != 'NM':
                sum += 1/6
    sum = sum/len(g)
    # weight of different attribute
    weight = [1, 1, 1, 1]
    loss_gender = [0, 1]
    loss_age = [0, 5/100, 10/100, 20/100, 1]
    loss_race = [0, 1]
    loss_marital_status_tree = [0, sum, 1]
    # calculate lm for all attribute
    LM = weight[0]*loss_age[vec[0]]+weight[1]*loss_gender[vec[1]] + \
        weight[2]*loss_race[vec[2]]+weight[3] * \
        loss_marital_status_tree[vec[3]]
    return g, True, LM


'''
function:func_vec
input:sum(total Generalization level)
      mi(max Generalization level for attribute i)
output: ans(all possiable Generalization vectors)
'''


def func_vec(sum, m1, m2, m3, m4):
    ans = []
    for i in range(m1+1):
        for j in range(m2+1):
            for k in range(m3+1):
                l = sum-i-j-k
                if l <= m4 and l >= 0:
                    ans.append([i, j, k, l])
    return ans


# main start from here
k = 10
MaxSup = 200
print("k=", k)
print("MaxSup=", MaxSup)
low = 0
high = 4+1+1+2
sol = []
start_time = time.time()
Best_LM = 4
while low < high:
    sol = []
    mid = int((low+high)/2)
    reach = False
    tmp = func_vec(mid, 4, 1, 1, 2)
    flag = False
    for i in range(len(tmp)):
        sol, reach, LM = satisfy(k, tmp[i], MaxSup)
        if reach == True:
            flag = True
            print("\nLM:", LM)
            print("Generalization vector:", tmp[i])
            if LM < Best_LM:
                Best_LM = LM
                ans = sol
                ans = pd.DataFrame(
                    ans, columns=['age', 'gender', 'race', 'marital_status', 'occupation'])
                ans.to_csv("ans.csv", index=False, sep=',')
            reach = False
    if flag == True:
        high = mid
    else:
        low = mid+1
end_time = time.time()
print("\nLM=%0.2f" % Best_LM)
print("Running time %0.2f" % (end_time - start_time) + " seconds")
