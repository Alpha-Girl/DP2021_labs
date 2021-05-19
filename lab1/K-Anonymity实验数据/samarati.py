from numpy.lib.function_base import average
import pandas as pd
import time
import csv
import math
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
        return str(tmp*m)+"-"+str(tmp*m+m-1)


'''
function:satisfy
input:  k (k-Anonymity),
        vec (generalization level for all attribute)
        MaxSup (max number of deleted records )
output: g (data after k-Anonymity),
        IsSatisfied (bool, True/Flase)
        LM (Loss Metric)
'''


def satisfy(k, vec, MaxSup, Utility_evaluation):
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
    LM=0
    if Utility_evaluation == 'Loss_Metric':
        # calculate lm for marital_status_treecd
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
    elif Utility_evaluation == 'Discernability_Metric':
        LM = 0
        tmp = list(d.keys())
        for j in range(len(tmp)):
            t = d.get(tmp[j])
            if t >= k:
                LM += t**2
        LM += l*count
    elif Utility_evaluation == 'entropy':
        LM = 0
        l = len(g)
        for i in range(4):
            d = {}
            for j in g:
                tmp = j[i]
                c = d.get(tmp)
                if c == None:
                    d[tmp] = 1
                else:
                    d[tmp] = c+1
            tmp = list(d.keys())
            for j in range(len(tmp)):
                t = d.get(tmp[j])
                if t >= k:
                    p = t/l
                    LM += p*math.log(p)

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
k = 20
MaxSup = 50
print("k=", k)
print("MaxSup=", MaxSup)
low = 0
high = 4+1+1+2
sol = []
start_time = time.time()
# Loss_Metric,Discernability_Metric,entropy,Just_do_it
Utility_evaluation = 'entropy'
Best_LM = float('inf')
Best_height = high
Best_vec = []
while low < high:
    sol = []
    mid = int((low+high)/2)
    reach = False
    tmp = func_vec(mid, 4, 1, 1, 2)
    flag = False
    for i in range(len(tmp)):
        sol, reach, LM = satisfy(k, tmp[i], MaxSup, Utility_evaluation)
        if reach == True:
            flag = True
            #print("\nLM:", LM)
            #print("Generalization vector:", tmp[i])
            # if LM < Best_LM:
            if mid < Best_height or (mid == Best_height and LM < Best_LM):
            #if True:
                Best_height = mid
                Best_LM = LM
                Best_vec = tmp[i]
                ans = sol
                ans = pd.DataFrame(
                    ans, columns=['age', 'gender', 'race', 'marital_status', 'occupation'])
                ans.to_csv("adult_samarati.csv", index=False, sep=',')
            reach = False
    if flag == True:
        high = mid
    else:
        low = mid+1
end_time = time.time()
print(Best_vec)
print("LM=%0.2f" % Best_LM)
print("Running time %0.2f" % (end_time - start_time) + " seconds")
print("Finish Samarati!!")
