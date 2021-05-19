# 数据隐私的方法伦理和实践Lab1实验报告

## 实验目的

1. 实现`Samarati`算法。
2. 实现`Mondrian`算法。
3. 测试不同的`k`和`MaxSup`对运行时间和LM的影响。
4. 调研并探究如何选择输出增加`Samarati`算法的可用性。
5. 增加`Mondrian`算法处理`categorical`类变量。

## 实验环境

1. PC一台
2. Windows操作系统
3. VSCode编辑器
4. Python 3.8.1

## 实验原理

### Samarati算法

#### 输入：

1. `k`：处理后的结果中每个相同的QI组合至少有k个元组。
2. `adult.data`：待泛化的数据集
3. `MaxSup`：可以删去的不满足k匿名的元组数。
4. `T`：泛化的层次结构。
5. `Utility_evaluation`：可用性评估方法。

#### 输出：

1. `Time`：算法运行时间。
2. `LM`：对应评估方法下的效用函数值。
3. `sol`：各个QI的泛化层次。
4. `adult_samarati.csv`：泛化的满足k匿名的数据集。

#### 伪代码：

```python
low = 0
high = height(T)
Best_LM = Inf
while low < high:
    mid = (low + high)/2
    flag = False
    for vec that sum(vec)=mid:
        data_samarati, reach, lm = satisfy(k,vec,MaxSup,Utility_evaluation)
        if reach && lm < Best_LM:
            data_anonymized = data_samarati
            sol = vec
            Best_LM = lm
            flag = True
	if flag == True:
        high = mid
	else:
        low = mid + 1
return data_anonymized, sol, Best_LM
```

注：`satisfy()`函数用于测试在给定`k`,`MaxSup`和`Utility_evaluation`的情况下，泛化层次是否满足k匿名，返回泛化后的数据集，是否满足，效用函数值。

### Mondrian算法

#### 输入：

1. `k`：处理后的结果中每个相同的QI组合至少有k个元组。
2. `adult.data`：待泛化的数据集

#### 输出：

1. `Time`：算法运行时间。
2. `LM`：Loss Metric。
3. `adult_mondrian.data`：泛化的满足k匿名的数据集。

#### 伪代码：

```
Anonymize(Partition)
	if(no allowable multidimensional cut for Partition)
		return Φ:Partition -> summary
	else
		dim <- choose_dimension()
		fs <- frequency_set(Partition,dim)
		splitVal <- find_median(fs)
		lhs <- {t∈Partition:t.dim ≤ splitVal}
		rhs <- {t∈Partition:t.dim ＞ splitVal}
		return Anonymize(rhs)∪Anonymize(lhs)
```

注：

1. `choose_dimension()`函数用于选取被泛化程度最大的维度。
2. `frequency_set()`函数用于求取对应维度上的取值分布。
3. `find_median()`函数用于根据分布求取中位数。
4. 类别型属性先化为数值型进行处理，在输出过程中再化为类别型即可。

### 评估方法

#### LM(Loss Metric)

LM的计算方式如下：

- 对于数值型属性，其LM=(U{i}-L{i})/U-L。
- 对于类别型属性，其LM=(M-1)/(A-1)，其中A为所有类别数，M为以该泛化节点为根节点的叶子节点数。
- 每个属性的LM为其所有泛化后元素的LM的平均值。
- 整体的LM为所有属性的LM 之和。

LM考虑了泛化层次带来的信息损失，但认为不同属性的信息量是相同的（这点可以考虑在求和时对，不同属性的LM加权来调整，默认均为1），而且没有考虑被删去的元组的损失。

####  DM(Discernability Metric)

DM的计算方式如下：

- 对于未被删除的元组，其损失为泛化后QI与其相同的元组数。
- 对于被删除的元组，其损失为数据集的大小。
- 整体的损失为所有元组的损失之和。

DM考虑了泛化带来的损失（同一等价类的元组数越多，损失越大），还考虑了被删除元组带来的损失，不过缺乏对泛化层次的影响的度量。

#### entropy

entropy的计算方式如下：

- 泛化后的数据集大小为`|D’|`
- 泛化后每个等价类（QI相同的为同一等价类）的大小为`|A|`，其损失为`plog(p)`（其中`p=|A|/|D’|`）

entropy参考信息熵，考虑了泛化后的数据集的信息量，信息量越大，损失越小。为了统一效用函数越小，表示效果越好，这里的熵未取负号。

## 实验结果

### 不同k，MaxSup的情况下的Samarati（未考虑Utility）

|  k   | MaxSup | vec[age,gender,race,marital_status] | Time(s) |
| :--: | :----: | :---------------------------------: | :-----: |
|  10  |  200   |            [1, 0, 1, 0]             |  18.65  |
|  10  |  100   |            [1, 0, 1, 1]             |  17.89  |
|  10  |   50   |            [3, 0, 1, 0]             |  14.06  |
|  5   |  200   |            [1, 0, 1, 0]             |  20.71  |
|  5   |  100   |            [1, 0, 1, 0]             |  16.89  |
|  5   |   50   |            [2, 0, 1, 0]             |  18.73  |
|  20  |  200   |            [1, 1, 1, 0]             |  20.31  |
|  20  |  100   |            [2, 1, 1, 0]             |  12.54  |
|  20  |   50   |            [1, 0, 1, 2]             |  11.16  |

#### 使用不同的评估方法

|  k   | MaxSup |        LM        |          DM          |      entropy      |
| :--: | :----: | :--------------: | :------------------: | :---------------: |
|  10  |  200   | 1.05[1, 0, 1, 0] | 39728101[1, 0, 1, 0] | -4.24[1, 0, 1, 0] |
|  10  |  100   | 1.16[1, 0, 1, 1] | 14746387[0, 0, 1, 2] | -4.57[0, 0, 1, 2] |
|  20  |  200   |  2[0, 0, 1, 2]   | 19107497[0, 0, 1, 2] | -4.55[0, 0, 1, 2] |
|  30  |  200   | 2.05[1, 0, 1, 2] | 27583737[0, 1, 1, 2] | -3.92[0, 1, 1, 2] |
|  20  |   50   | 2.05[1, 0, 1, 2] | 23809399[0, 1, 1, 2] | -3.93[0, 1, 1, 2] |

由表中数据可以看出：

1. 在相同的`k`和`MaxSup`下，不同评价指标的输出结果（泛化层次向量）并不相同，比如`k=10`，`MaxSup=100`时。
2. 在不同评价指标下，效用最高时的`k`和`MaxSup`也不相同。
3. `DM`和`entropy`的结果较为接近。
4. `LM`在`k=30`，`MaxSup=200`和`k=20`，`MaxSup=50`时的效用值一样，但`k=20`，`MaxSup=50`的所含的信息量较大，而`LM`却无法体现。

### Mondrian

|  k   |  LM  | Time(s) |
| :--: | :--: | :-----: |
|  10  | 0.77 |  0.42   |
|  20  | 0.93 |  0.31   |
|  40  | 1.15 |  0.30   |
|  80  | 1.46 |  0.25   |
| 160  | 1.99 |  0.22   |
| 320  | 2.88 |  0.22   |

由表中数据可以看出：

1. `Mondrian`算法的`LM`随`k`的增大逐渐减小，表明泛化程度越大，可用性越差，与理论相符。
2. `Mondrian`算法的运行时间随`k`的增大而减小，符合实验预期。

