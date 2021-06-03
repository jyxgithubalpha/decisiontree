import numpy as np
import matplotlib.pyplot as plt

class Node():
    def __init__(self):
        self.type = 0
        self.fcs = 0
        self.slices = []
        self.father = None
        self.sonnum = 0
        self.son = []
        self.flag = 0
        self.contain = [0,0,0]
        self.ct = 0
        self.cT = 0
        self.g = 0
'''
一个节点的定义：
type是最后的分类结果
fcs是featurechoose的简写,就是这个节点要选择哪个特征
slices是分片的意思，也就是说选择的特征的分区阈值是多少，是一个数组
father指向父节点
sonnum是number of son的简称，就是有多少个子节点
son是一个指向子节点的数组
flag用来标记这个节点有没有走到过
contain是这个节点各个结果各自包含了多少个，如果1号花10朵，2号花8朵，3号花11朵，那么就是【10 8 11】
ct是这个节点的信息熵，cT是这个节点作为一个树的信息熵，用来后剪枝用
g是(ct-cT)/(|T|-1)，也是用来后剪枝用
'''


def gennode(father):
    son = Node()
    son.father = father
    father.sonnum = father.sonnum + 1
    father.son.append(son)
    return son
'''
生成一个节点
'''

def datatype(data):
    flag1, flag2, flag3 = 0, 0, 0
    for line in data:
        if line[5] == 1:
            flag1 = flag1 + 1
        if line[5] == 2:
            flag2 = flag2 + 1
        if line[5] == 3:
            flag3 = flag3 + 1
    return 1 if (flag1 != 0 and flag2 == flag3 == 0) \
        else 2 if (flag2 != 0 and flag1 == flag3 == 0) \
        else 3 if (flag3 != 0 and flag1 == flag2 == 0) \
        else 0 if (flag1 == flag2 == flag3 == 0) \
        else 11 if (max(flag1, flag2, flag3) == flag1) \
        else 12 if (max(flag1, flag2, flag3) == flag2) \
        else 13 if (max(flag1, flag2, flag3) == flag3) \
        else 99
'''
判断这个样本中的种类情况：
如果都是同一个标签，就返回标签指
如果是不同的，就返回其中最大的标签值+10，这样以后可以判断返回的是都是同一个标签还是标签最多的
'''


def gain(data, featureindex, slicesindex):
    parts = len(slicesindex) + 1
    slicescnt = [0 for i in range(parts)]
    for line in data:
        if line[featureindex + 1] < slicesindex[0]:
            slicescnt[0] = slicescnt[0] + 1
        elif line[featureindex + 1] >= slicesindex[parts - 2]:
            slicescnt[parts - 1] = slicescnt[parts - 1] + 1
        else:
            for i in range(parts - 2):
                if slicesindex[i] <= line[featureindex + 1] < slicesindex[i + 1]:
                    slicescnt[i + 1] = slicescnt[i + 1] + 1

    entd = 0
    sum = 0
    gain = 0
    for i in range(parts):
        sum = sum + slicescnt[i]
    for i in range(parts):
        p = slicescnt[i] / sum
        if p == 0:
            continue
        entd = entd - (p * np.log2(p) if p != 0 else 0)
        gain = gain - p * (p * np.log2(p) if p != 0 else 0)
    gain = gain + entd
    return gain
'''
计算增益
data是输入的样本
featureindex是选择了哪个特征算增益，是index of feature的缩写
sliceindex是选择的特征的分区情况，是index of slices的缩写
'''


def featurechoose(data, feature, slices):
    gainlist = []
    for i in range(len(feature)):
        gainlist.append(gain(data, i, slices[i]))
    if len(set(gainlist)) == 1:
        return -1
    else:
        return np.argmax(gainlist)
'''
是choose feature的缩写，就是根据增益选择哪一个作为决策树节点要判断的特征
data是输入的数据
feature是特征的集合
slices是特征的分区的集合
'''

def datachoose(data, fcs, slicesindex, i):
    newdata = []
    for line in data:
        if i == 0 and line[fcs + 1] < slicesindex[0]:
            newdata.append(line)
        elif i == len(slicesindex) and line[fcs + 1] >= slicesindex[len(slicesindex) - 1]:
            newdata.append(line)
        elif slicesindex[i - 1] <= line[fcs + 1] < slicesindex[i]:
            newdata.append(line)
    return newdata
'''
选择特征后（fcs是选择的特征），根据要选择的分区（slicesindex是特征的分区，i代表要选择哪个分区），获得满足分区条件的样本
'''

def gentree(data, feature, slices):
    root = Node()
    dtp = datatype(data)
    if dtp == 1:
        root.type = 1
        return root
    if dtp == 2:
        root.type = 2
        return root
    if dtp == 3:
        root.type = 3
        return root

    fcs = featurechoose(data, feature, slices)
    if len(feature) == 0 or fcs == -1:
        root.type = dtp - 10
        return root

    root.fcs = fcs
    root.slices = slices[fcs]
    for i in range(len(slices[fcs]) + 1):
        lastdata = data[:]
        lastfeature = feature[:]
        lastslices = slices[:]
        lastfcs = fcs
        data = datachoose(data, fcs, slices[fcs], i)
        feature.pop(fcs)
        slices.pop(fcs)
        if len(data) == 0:
            newnode = gennode(root)
            newnode.type = dtp - 10
            data = lastdata
            feature = lastfeature
            slices = lastslices
            fcs = lastfcs
        else:
            newnode = gentree(data, feature, slices)
            newnode.father = root
            root.son.append(newnode)
            root.sonnum = root.sonnum + 1
            data = lastdata
            feature = lastfeature
            slices = lastslices
            fcs = lastfcs
    return root
'''
生成决策树
采用了递归的算法，具体算法见报告
'''

def prediction(input, tree, slices):
    if tree.type != 0:
        return tree.type
    else:
        for i in range(len(slices[tree.fcs]) + 1):
            if input[tree.fcs + 1] < slices[tree.fcs][0] and i == 0:
                return prediction(input, tree.son[0], slices)
            elif input[tree.fcs + 1] >= slices[tree.fcs][0] and i == len(slices[tree.fcs]):
                return prediction(input, tree.son[len(slices[tree.fcs])], slices)
            elif slices[tree.fcs][i - 1] <= input[tree.fcs + 1] <= slices[tree.fcs][i]:
                return prediction(input, tree.son[i], slices)
'''
输入一个数据（input）后，根据决策树（tree）和分区情况（slcies）预测结果，仍旧是递归的算法
'''

def datalinedeal(dataline):
    dataline[0] = int(dataline[0])
    dataline[1] = float(dataline[1])
    dataline[2] = float(dataline[2])
    dataline[3] = float(dataline[3])
    dataline[4] = float(dataline[4])
    dataline[5] = 1 if dataline[5] == 'setosa' else 2 if dataline[5] == 'versicolor' else 3
    return dataline
'''
洗数据，把txt文件中作为字符串输入的数据洗成我们需要的数据格式，并将label变为整数1 2 3便于之后的使用
'''

def datadevide(dataset, indexrange):
    traindata = []
    testdata = []
    for i in range(int(len(indexrange) / 2)):
        for line in dataset:
            if indexrange[2 * i] <= line[0] <= indexrange[2 * i + 1]:
                traindata.append(line)
            elif 2 * i + 1 == len(indexrange)-1 and indexrange[2 * i + 1] < line[0]:
                testdata.append(line)
            elif indexrange[2 * i + 1] < line[0] < indexrange[2 * i + 2]:
                testdata.append(line)
    return [traindata,testdata]
'''
划分数据集为训练集和测试集，采用分层采样的方式
'''

def fccy(start, end, n, nsep):
    result = []
    for i in range(int((end - start + 1) / (n + nsep))):
        result.append(start + i * (n + nsep))
        result.append(start + i * (n + nsep) + n-1)
    return result
'''
fccy是分层采样的首字母的缩写，fccy主要是通过起始坐标start，终止坐标end，
选择作为训练集的一波样本数n和作为测试机的一波样本数nsep（sep是间断的意思，就是训练集的间断，也就是测试集）
'''


def rightrate(testdata,Mytree,slices):
    sum = 0
    right = 0
    wrong = 0
    for line in testdata:
        if prediction(line, Mytree, slices) == line[5]:
            right = right + 1
            sum = sum + 1
        else:
            wrong = wrong + 1
            sum = sum + 1
    print(right, wrong, right / (right + wrong))
    return right / (right + wrong)
'''
正确率，就是训练出来的决策树Mytree对于测试集合testdata和分区情况slices的正确率，返回正确数，错误数和正确率
'''

def loadtreecontain(tree,slices,testdata):
    if tree.sonnum == 0:
        for line in testdata:
            tree.contain[int(line[5])-1] = tree.contain[int(line[5])-1] + 1
    else:
        for line in testdata:
            tree.contain[int(line[5])-1] = tree.contain[int(line[5])-1] + 1
#            tree.contain[prediction(line,tree,slices)-1] = tree.contain[prediction(line,tree,slices)-1] + 1
        for i in range(tree.sonnum):
            loadtreecontain(tree.son[i],slices,datachoose(testdata, tree.fcs, slices[tree.fcs], i))
'''
load contain of tree的缩写，就是将每个节点中样本(testdata)的结果作为数组填充到节点内部。
比如说，这个节点下有8个A，5个B，9个C，就会在node.contain中填充【8 5 9】
目的是为了计算后剪枝的ct和CT
'''

def c(tree,flag):
    if flag == 0:
        C = 0
        Nt = sum(tree.contain)
        for x in tree.contain:
            C = C - ((x * np.log10(x / Nt)) if x != 0 else 0)
        return C
    else:
        if tree.sonnum == 0:
            C = 0
            Nt = sum(tree.contain)
            for x in tree.contain:
                C = C - ((x*np.log10(x/Nt)) if x != 0 else 0)
            return C
        else:
            return sum(c(x,1) for x in tree.son)
'''
计算剪枝的ct和CT,flag为0计算ct，为1计算cT，小写的t代表节点，大写的T代表树
'''

def loadtreecandg(tree):
    if tree.sonnum == 0:
        tree.ct = c(tree, 0)
        tree.cT = c(tree, 1)
        tree.g = (tree.ct - tree.cT) / (tree.sonnum - 1)
        return
    else:
        tree.ct = c(tree,0)
        tree.cT = c(tree,1)
        tree.g = (tree.ct-tree.cT)/(tree.sonnum - 1)
        for x in tree.son:
            loadtreecandg(x)
'''
把计算的ct，cT和g装入树里对应的节点，方便以后后剪枝用
'''

def findalpha(tree,alpha):
    alpha = tree.g if tree.g <= alpha and tree.type == 0 else alpha
    for x in tree.son:
        alpha = findalpha(x,alpha) if findalpha(x,alpha) <= alpha else alpha
    return alpha
'''
找到alpha，alpha就是数中最小的g，对这个节点实现剪枝
通常来说最小的g意味着这个节点中主要为某一种结果，比如说15个A，1个B，2个C
这时候采用剪枝将这个节点全部变为A可以提高决策树的泛化能力
'''

def findnode(node,alpha):
    if node.g == alpha and node.type == 0:
        return node
    elif node.g != alpha and node.sonnum == 0:
        return None
    else:
        for x in node.son:
            if findnode(x,alpha) != None:
                return findnode(x,alpha)
'''
找到alpha对应的节点，也就是要实施剪枝的节点
'''

def treepruning(tree):
    if tree.father == None and tree.sonnum == 0:
        return
    alpha = findalpha(tree,9999)
    node = findnode(tree,alpha)
    node.sonnum = 0
    node.type = np.argmax(node.contain) + 1
'''
pruning of tree的缩写，就是剪枝的意思。
找到需要剪枝的节点，然后这个节点采用多数表决的原则
'''

with open(r'C:\Users\meaic\Desktop\pml\Iris\iris.txt', 'r') as f:
    dataset = []
    for line in f:
        if line[1] < '0' or line[1] > '9':
            continue
        line = line.split()
        dataset.append(datalinedeal(line))
    dataset = np.array(dataset)
feature = [1, 2, 3, 4]
slices = [[5, 6, 7], [2, 3, 4], [1, 3, 5], [1, 2, 3]]
[traindata,testdata] = datadevide(dataset, fccy(1,150,10,40))
Mytree = gentree(traindata, feature, slices)
'''
读取txt文件，洗数据并划分训练集和测试集，输入我们的特征和分区情况，然后得到我们需要的决策树
'''



feature = [1, 2, 3, 4]
slices = [[5, 6, 7], [2, 3, 4], [1, 3, 5], [1, 2, 3]]
rightrate(testdata,Mytree,slices)
'''
获得我们的决策树的正确率
'''

correctrate = []
loadtreecontain(Mytree,slices,dataset)
loadtreecandg(Mytree)
for i in range(10):
    treepruning(Mytree)
    correctrate.append(rightrate(testdata, Mytree, slices))
fig = plt.figure(0)
ax = fig.add_subplot(111)
ax.plot(correctrate)
plt.show()
'''
连续剪枝并输出正确率，不难发现正确率会先上升后下降，
这是一个从过拟合到欠拟合的过程，最后会变成1/3也就是样本的先验概率，欠拟合无穷大的结果
'''



#以下为画图部分
def getdepth(tree):
    if tree.type != 0:
        return 1
    else:
        return 1 + max(getdepth(x) for x in tree.son)
'''
获得树的最大深度，为安排图片的位置做准备
'''
def getleaf(tree):
    if tree.type != 0:
        return 1
    else:
        return sum(getleaf(x) for x in tree.son)
'''
获得树的叶节点树，为安排图片的位置做准备
'''
decisionnode = dict(boxstyle='sawtooth', fc='0.8')
leafnode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='->')
'''
定义分支节点，叶节点和箭头的样式
'''


def plotnode(txt,now,next,nodetype,ax):
    ax.annotate(txt, xy=next,xytext=now,bbox=nodetype, arrowprops=arrow_args)
'''
画一个节点，内容为txt，样式为nodetype，位置为now，再画一个箭头，从now指向next
'''


def gettxt(node):
    type = 'root' if node.father == None else 'leaf' if node.type !=0 else 'branch'
    result = str(node.type) if node.type != 0 else 'None'
    fcs = 'None' if node.type != 0 else str(int(node.fcs))
    slices = str(node.slices) if node.type == 0 else 'None'
    return 'type:' + type + '\nresult:' + result + '\nfcs:' + fcs +'\nslices:'+ slices
'''
处理我们的决策树图形中节点显示的内容，包括：
节点的类型（根节点root，分支节点branch还是叶节点leaf）
节点的结果，叶绩点的result代表最后分类的标签
节点选择的属性，fcs就是上面说的要选择的属性
节点的分区情况，比如【1,3,5】代表这个节点会根据选择选择的属性分为4块，x<1,1<=x<3,3<=x<5,x>5四部分，然后生成四个节点
'''


def plottree(tree,now,ax,depth,sumleaf):
    if tree.type != 0:
        plotnode(gettxt(tree),now,now,leafnode,ax)
    else:
        sonleaf = []
        for i in range(tree.sonnum):
            sonleaf.append(getleaf(tree.son[i]))
        for i in range(tree.sonnum-1,-1,-1):
            if i == 0:
                sonleaf[i] = 0
            else:
                sonleaf[i] = sum(sonleaf[k] for k in range(i))
        for i in range(tree.sonnum):
            plotnode(gettxt(tree),now,[now[0]+sonleaf[i]/sumleaf,now[1]-1/depth],decisionnode,ax)
            plottree(tree.son[i],[now[0]+sonleaf[i]/sumleaf,now[1]-1/depth],ax,depth,sumleaf)
'''
决策图画图，运用了递归的算法。
如果是叶节点就只画一个节点
如果是别的节点就画节点和这个节点的下指箭头，然后在子节点的位置继续调用这个函数
节点位置now，坐标系ax，根节点tree，depth最大深度，sumleaf叶节点个数，
'''

fig = plt.figure(1,facecolor='white',figsize=(20,20))
ax = fig.add_subplot(111,frameon = False)
depth = getdepth(Mytree)
sumleaf = getleaf(Mytree)
plottree(Mytree,[0,1],ax,depth,sumleaf)
plt.show()
'''
画图
'''