import pandas as pd

import csv
raw_data = []
with open('test.csv')as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        raw_data.append(row)
s_data = []
with open('test.csv')as ss:
    ss_csv = csv.reader(ss)
    for row in ss_csv:
        s_data.append(row)

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


def satisfy(k, vec, MaxSup):
    t = raw_data
    g = []
    g = s_data
    l = len(t)
    d = {}
    for i in range(l):
        g[i][0] = age_Generalization(t[i][0], vec[0])
        g[i][1] = Generalization(gender_tree, t[i][1], vec[1])
        g[i][2] = Generalization(race_tree, t[i][2], vec[2])
        g[i][3] = Generalization(marital_status_tree, t[i][3], vec[3])
        x = str(g[i][0])+str(g[i][1])+str(g[i][2])+str(g[i][3])
        tmp = d.get(x)
        if tmp == None:
            d[x] = 1
        else:
            d[x] = tmp+1
    count = 0
    tmp = list(d.keys())
    for j in range(len(tmp)):
        if d.get(tmp[j]) < k:
            count = count+d.get(tmp[j])
        if count > MaxSup:
            return [], False
    print("ok")
    return g, True


def func_vec(sum, m1, m2, m3, m4):
    ans = []
    for i in range(m1+1):
        for j in range(m2+1):
            for k in range(m3+1):
                l = sum-i-j-k
                if l <= m4 and l >= 0:
                    ans.append([i, j, k, l])
    return ans


k = 10
MaxSup = 200
low = 0
high = 4+1+1+2
sol = []
while low < high:
    sol = []
    mid = int((low+high)/2)
    reach = False
    tmp = func_vec(mid, 4, 1, 1, 2)
    for i in range(len(tmp)):
        sol, reach = satisfy(k, tmp[i], MaxSup)
        if reach == True:
            print("v:")
            print(tmp[i])
            ans = sol
            ans = pd.DataFrame(
                ans, columns=['age', 'gender', 'race', 'marital_status', 'occupation'])
            ans.to_csv("ans.csv", index=False, sep=',')
            break
    if reach == True:
        high = mid
    else:
        low = mid+1
print(low)
print(high)
