from sklearn.ensemble import RandomForestClassifier
import numpy as np
import math


domainlist = []
domainlist2= []

class Domain:
    def __init__(self,_name,_label):
    self.name = _name
    self.label = _label

    def returnData(self):
        return [len(self.name),countnumber(self.name),cal_entropy(self.name)]

    def returnLabel(self):
        if self.label == "notdga":
	          return 0
        else:
            return 1

def countnumber(string):
    int_count=0
    for i in string:
        if i.isdigit():
            int_count +=1

    return int_count

def cal_entropy(text):
    h = 0.0
    sum = 0
    letter = [0] * 26
    text = text.lower()
    for i in range(len(text)):
        if text[i].isalpha():
            letter[ord(text[i]) - ord('a')] += 1
            sum += 1
    for i in range(26):
        p = 1.0 * letter[i] / sum
        if p > 0:
            h += -(p * math.log(p, 2))
    return h

def initData(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
	      if line.startswith("#") or line =="":
	          continue
	      tokens = line.split(",")
	      name = tokens[0]
	      label = tokens[1]
	      domainlist.append(Domain(name,label))
			
def initData2(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
        if line.startswith("#") or line =="":
	          continue
        tokens = line.split(",")
	      name = tokens[0]
	      domainlist2.append(name)

def main():
    initData("train.txt")
    initData2("test.txt")
    featureMatrix = []
    labelList = []
    for item in domainlist:
        featureMatrix.append(item.returnData())
        labelList.append(item.returnLabel())

    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix,labelList)
    
    arr=["notdga","dga"]
    f=open("result.txt",'w')
    for item in domainlist2:
        t=clf.predict([[len(item),countnumber(item),cal_entropy(item)]])
        f.write(item+','+np.array(arr)[t][0]+'\n')
		                        

if __name__ == '__main__':
    main()
