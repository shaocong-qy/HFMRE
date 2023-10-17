# import torch
# import torch.nn.functional as F
# from torch import nn
#
# vec1 = torch.rand(3,768)
# vec2 = nn.Parameter(torch.empty(58,768))
# nn.init.xavier_normal(vec2)
# cos_sim = F.cosine_similarity(vec1, vec2, dim=0)
# print(cos_sim)
import numpy as np
import torch
import model
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
class HuffmanTreeNode:
    def __init__(self,
                 patt=None,
                 root=None,
                 value: str = None,
                 frq: int = 0,
                 feature=None,
                 weight=0
                 ) -> None:
        self.root = root
        self.value = value
        self.frq = frq
        self.weight= weight
        self.feature= feature
        self.left = None
        self.right = None
        self.size = 1
        self.softmax=nn.Softmax(-1)
        self.patt=patt
    # def Setweight(self, nodelist):
    #     frqs=[]
    #     for i in range(len(nodelist)):
    #         frqs.append(nodelist[i].frq)
    #         # if (i==0):
    #         #     frqs=nodelist[i].frq
    #         # else:
    #         #     frqs=torch.stack((frqs,nodelist[i].frq))
    #     frqs = torch.stack(frqs)
    #     weights=self.softmax(frqs)
    #     for i in range(len(nodelist)):
    #         nodelist[i].weight=weights[i]
    #     return self



    def Set_feature_frq_weight(self, left,right,nodelist):

        feature = torch.cat((right.feature,left.feature),dim=-1)
        # feature = (right.feature+left.feature)/2
        if len(nodelist)==0:
            Act=True
        else:
            Act=False
        feature = model.ModelRQuery.FC1(self=self.patt, x=feature, isAct=Act)
        self.feature=feature
        features=[]

        features.append(feature)
        for i in range(len(nodelist)):
            features.append(nodelist[i].feature)

        features=torch.stack(features)
        distances=F.cosine_similarity(features.unsqueeze(1),features.unsqueeze(0), dim=-1)
        distances=distances.sum(axis=0)
        weights = self.softmax(distances)
        self.frq=distances[0]
        self.weight=weights[0]
        for i in range(len(nodelist)):
            nodelist[i].frq = distances[i+1]
            nodelist[i].weight = weights[i + 1]


        return self


    def Setleft(self, left):
        self.left = left
        # self.frq += left.Getfrq()
        self.size += left.GetSize()
        return self

    def Setright(self, right):
        self.right = right
        # self.frq += right.Getfrq()
        self.size += right.GetSize()
        return self

    def Getfrq(self):
        return self.frq

    def Getfrq2(self):
        x=self.frq
        x =x.detach().cpu()
        x=round(float(x),2)
        return x

    def Getvalue(self):
        return self.value

    def GetSize(self):
        return self.size

    def Hasright(self):
        return self.right

    def Hasleft(self):
        return self.left

    def Isroot(self):
        return self.root

    def __str__(self) -> str:
        if self.root:
            return f'root, sum of frequency:{self.frq}'
        else:
            return f'value: {self.value}, frequency: {self.frq}'


class huffmanTree:
    def __init__(self,patt,num) -> None:
        # 对字典按照其values进行排序
        self.num=sorted(num, key=lambda x: x[1], reverse=False)
        self.list = []  # 一个储存列表
        self.coding = {}  # 编码结果
        self.patt = patt
        self.buildHuffmanTree()
        self._iter_node(self.list[0])

    def buildHuffmanTree(self):
        self.list = [HuffmanTreeNode(patt=self.patt,root=False, value=i[0], frq=i[1], feature=i[2], weight=i[3]) for i in self.num]

        if (len(self.list)) == 1:
            self.list[0].feature = model.ModelRQuery.FC2(self=self.patt, x=self.list[0].feature, isAct=True)

        while len(self.list) > 1:
            # 将两个小的节点合并  小的放左边
            right_node = self.list[1]
            left_node = self.list[0]
            # 注意pop顺序
            self.list.pop(1)
            self.list.pop(0)
            temp_node = HuffmanTreeNode(patt=self.patt,root=True)
            temp_node.Setright(right_node)
            temp_node.Setleft(left_node)
#################################################
            temp_node.Set_feature_frq_weight(left_node,right_node,self.list)

#################################################
            # 将合并后的根节点放回list中
            if len(self.list) == 1:
                if temp_node.Getfrq() < self.list[0].Getfrq():
                    self.list.insert(0, temp_node)
                    # temp_node.Setweight(self.list)
                else:
                    self.list.insert(1, temp_node)
                    # temp_node.Setweight(self.list)
            elif len(self.list) == 0:
                self.list.insert(0, temp_node)
                # temp_node.Setweight(self.list)
            else:
                for i in range(len(self.list) - 1):
                    if i == 0 and temp_node.Getfrq() <= self.list[i].Getfrq():
                        self.list.insert(i, temp_node)
                        # temp_node.Setweight(self.list)
                        continue
                    elif self.list[i].Getfrq() < temp_node.Getfrq() <= self.list[i + 1].Getfrq():
                        self.list.insert(i + 1, temp_node)
                        # temp_node.Setweight(self.list)
                        continue
                    elif i == len(self.list) - 2 and temp_node.Getfrq() > self.list[i + 1].Getfrq():
                        self.list.insert(i + 2, temp_node)
                        # temp_node.Setweight(self.list)
                        continue

    def _iter_node(self, node, code=''):
        if node:
            if not node.Isroot():
                self.coding[node.Getvalue()] = code
            self._iter_node(node.Hasleft(), code='0' + code)
            self._iter_node(node.Hasright(), code='1' + code)

    def getCode(self):
        return self.coding


    def getTree(self):
        return self.list[0]

class Draw_RBTree:
    def __init__(self, tree):
        self.tree = tree

    def show_node(self, node, ax, height, index, font_size):
        if not node:
            return
        x1, y1 = None, None
        if node.left:
            x1, y1, index = self.show_node(node.left, ax, height-1, index, font_size)
        x = 100 * index - 50
        y = 100 * height - 50
        if x1:
            plt.plot((x1, x), (y1, y), linewidth=2.0,color='b')
        circle_color = 'mediumspringgreen'
        text_color = 'black'
        ax.add_artist(plt.Circle((x, y), 50, color=circle_color))
        text = str(node.Getfrq2()) if node.Isroot() else node.Getvalue() + '\n' + str(node.Getfrq2())
        ax.add_artist(plt.Text(x, y, text, color= text_color, fontsize=font_size, horizontalalignment="center",verticalalignment="center"))
        # print(str(node.val), (height, index))

        index += 1
        if node.right:
            x1, y1, index = self.show_node(node.right, ax, height-1, index, font_size)
            plt.plot((x1, x), (y1, y), linewidth=2.0, color='b')

        return x, y, index

    def show_hf_tree(self, title):
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        left, right = self.get_left_length(), self.get_right_length(),
        height = 2 * np.log2(self.tree.size + 1)
        # print(left, right, height)
        plt.ylim(0, height*100 + 50)
        plt.xlim(0, 100 * self.tree.size + 100)
        self.show_node(self.tree, ax, height, 1, self.get_fontsize())
        plt.axis('off')
        plt.title(title)
        plt.show()

    def get_left_length(self):
        temp = self.tree
        len = 1
        while temp:
            temp = temp.left
            len += 1
        return len

    def get_right_length(self):
        temp = self.tree
        len = 1
        while temp:
            temp = temp.right
            len += 1
        return len

    def get_fontsize(self):
        count = self.tree.size
        if count < 10:
            return 30
        if count < 20:
            return 20
        return 16


if __name__ == '__main__':

    # num = {'a':10,'b':15,'c':12,'d':3,'e':4,'f':13,'g':1}

    num=[('a',10,torch.rand(1,3)),('b',15,torch.rand(1,3)),('c',12,torch.rand(1,3)),('d',3,torch.rand(1,3)),('e',4,torch.rand(1,3)),('f',13,torch.rand(1,3))]

    h = huffmanTree(num)
    tree = h.getTree()
    d = Draw_RBTree(tree)
    d.show_hf_tree('HuffmanTree')
    print(h.getCode())

