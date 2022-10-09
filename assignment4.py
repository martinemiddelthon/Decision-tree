# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 15:10:08 2022

@author: marti
"""

import pandas as pd
import numpy as np
import random
import copy
import graphviz

# Read in train and test data
attrNames = ['Attr0', 'Attr1', 'Attr2', 'Attr3', 'Attr4', 'Attr5', 'Attr6', 'Class']
traindf = pd.read_csv('train.csv', names = attrNames)
testdf = pd.read_csv('test.csv', names = attrNames)



class treeNode():

    def __init__(self, parent, examples, attributesRemaining):
        
        self.attribute = None
        self.parent = parent
        self.examples = examples
        self.attributesToAssess = attributesRemaining
        
        self.leftChild = None           # Child with attribute value 1
        self.rightChild = None          # Child with attribute value 2
        
        self.isLeftChild = False
        self.isRightChild = False
        
        self.isLeaf = False
        self.classifier = None
        
        
     
    def isLeaff(self):
        if self.isLeaf:
            return True
        return False
    
    def getClassifier(self):
        return str(self.classifier)
    
    def getClassifierInt(self):
        return self.classifier
    
    def getAttribute(self):
        return self.attribute
    
    def getLeftChild(self):
        return self.leftChild
    
    def getRightChild(self):
        return self.rightChild
    
    def getExamples(self):
        return self.examples
    
    def setLeftChild(self):
        self.isLeftChild = True
    
    def setRightChild(self):
        self.isRightChild = True

    
    
    def pluralityValue(self, examples):

        a = len(examples.loc[examples['Class'] == 1].index.tolist())
        b = len(examples.loc[examples['Class'] == 2].index.tolist())
        
        if a==0 and b>0:
            return 2
        
        elif a>0 and b==0:
            return 1
        
        else:
            numClasses = examples['Class'].value_counts()
            if numClasses[1] > numClasses[2]:
                return 1
            elif numClasses[1] < numClasses[2]:
                return 2
            else:
                r = random.random()
                if r < 0.5:
                    return 1
                elif r >= 0.5:
                    return 2
    
    
    def importanceRandom(self, attribute, examples):
        return random.random()
    
    def Entropy(self, examples):
        a = len(examples.loc[examples['Class'] == 1])   #number of class=1 in examples
        b = len(examples.loc[examples['Class'] == 2])   #number of class=2 in examples
        if (a==0 and b>0) or (a>0 and b==0):
            return 0
        
        vals = examples['Class'].value_counts()
        q = vals[1] / (vals[1] + vals[2])
        B = - (q * np.log2(q) + (1-q) * np.log2(1-q))
        return B
    
    def importanceEntropy(self, attributeString, examples):
        ex1 = examples.loc[examples[attributeString] == 1]
        ex2 = examples.loc[examples[attributeString] == 2]
        prop = len(ex1) / len(examples)
        
        Gain = self.Entropy(examples) - prop * self.Entropy(ex1) - (1-prop) * self.Entropy(ex2)
    
        return Gain

    
    def argmaxImportance(self, attributesList, examples, importanceFnc):
        numAttributes = len(attributesList)
        importanceVals = np.empty(numAttributes)

        for i in range(numAttributes):
            importanceVals[i] = importanceFnc(attributesList[i], examples)
        
        maxIndex = np.argmax(importanceVals)
        
        return attributesList[maxIndex]
    
    
    def learnDecisionTree(self, importance):
        ex = self.examples.copy()
        
        if len(ex) == 0:
            self.isLeaf = True
            self.classifier = self.pluralityValue(self.parent.getExamples())
            return 
        
        a = len(ex.loc[ex['Class'] == 1])   #number of class=1 in examples
        b = len(ex.loc[ex['Class'] == 2])   #number of class=2 in examples
        if (a==0 and b>0) or (a>0 and b==0):
            self.isLeaf = True
            if a==0:
                self.classifier = 2
            elif b==0:
                self.classifier = 1
            
            return 
        
        elif len(self.attributesToAssess) == 0:
            self.isLeaf = True
            self.classifier = self.pluralityValue(ex)
            return 
        
        if importance == 'Entropy':
            attributeString = self.argmaxImportance(self.attributesToAssess,
                                                        ex, self.importanceEntropy)
        elif importance == 'Random':
            attributeString = self.argmaxImportance(self.attributesToAssess,
                                                        ex, self.importanceRandom)
            
        self.attribute = attributeString
        attributesList = copy.deepcopy(self.attributesToAssess)
        attributesList.remove(self.attribute)
        
        exs1 = ex.loc[ex[attributeString] == 1]
        exs2 = ex.loc[ex[attributeString] == 2]
        
        childNode = treeNode(self, exs1, attributesList)
        childNode.setLeftChild()
        childNode.learnDecisionTree(importance)
        self.leftChild = childNode
        
        childNode2 = treeNode(self, exs2, attributesList)
        childNode2.setRightChild()
        childNode2.learnDecisionTree(importance)
        self.rightChild = childNode2
        
        return 
    
def predictDatapoint(datapoint, treeRoot):
        
    node = treeRoot
    attr = node.getAttribute()
    clas = 0
    
    if attr != None:
        attrValue = datapoint[attr].tolist()[0]
        if attrValue == 1:
            newNode = node.getLeftChild()
            clas = predictDatapoint(datapoint, newNode)
        elif attrValue == 2:
            newNode = node.getRightChild()
            clas = predictDatapoint(datapoint, newNode)
        
    else:
        clas = node.getClassifierInt()
        
    return clas 
        
 
def predictDataset(dataset, treeRoot):
    vals = np.empty(len(dataset))
    
    for i in range(len(dataset)):
        datapoint = dataset.iloc[[i]]
        vals[i] = predictDatapoint(datapoint, treeRoot)
    
    return vals


def successRate(dataset, treeRoot):
    actualvals = dataset['Class'].tolist()
    predictvals = predictDataset(dataset, treeRoot)
    
    accuracy = 0
    for i in range(len(actualvals)):
        if actualvals[i] == predictvals[i]:
            accuracy += 1
    
    accuracy = accuracy / len(actualvals) * 100
    
    return accuracy
    
    
counter = 0 
        
def visualizeTree(Node, graph, parentname=None): 
    global counter
    if parentname == None:
        parentname = str(counter)
        counter += 1
        graph.node(name = parentname, label = Node.getAttribute())
    
    name1 = str(counter)
    counter += 1
    name2 = str(counter)
    counter += 1
    
    child1 = Node.getLeftChild()
    if child1.isLeaff():
        graph.node(name = name1, label = child1.getClassifier())
        graph.edge(parentname, name1, label = '1')
    else:
        graph.node(name = name1, label = child1.getAttribute())
        graph.edge(parentname, name1, label = '1')
        visualizeTree(child1, graph, name1)
    
    
    child2 = Node.getRightChild()
    if child2.isLeaff():
        graph.node(name = name2, label = child2.getClassifier())
        graph.edge(parentname, name2, label = '2')
    else:
        graph.node(name = name2, label = child2.getAttribute())
        graph.edge(parentname, name2, label = '2')
        visualizeTree(child2, graph, name2)   


# ENTROPY

# Build tree
attributes = ['Attr0', 'Attr1', 'Attr2', 'Attr3', 'Attr4', 'Attr5', 'Attr6']
initRoot = treeNode(None, traindf, attributes)
initRoot.learnDecisionTree('Entropy')

# Visualize tree
graph = graphviz.Digraph(name = 'Decision tree', filename = 'desiciontree.pdf') 
visualizeTree(initRoot, graph)
graph.render(view=True)

# Evaluate on test set
a1 = successRate(testdf, initRoot)

print('\nENTROPY:\nSuccess rate: ',a1,'%')

# RANDOM

# Build tree
attributes = ['Attr0', 'Attr1', 'Attr2', 'Attr3', 'Attr4', 'Attr5', 'Attr6']
initRoot2 = treeNode(None, traindf, attributes)
initRoot2.learnDecisionTree('Random')

# Visualize tree
graph2 = graphviz.Digraph(name = 'Random decision tree', filename = 'desiciontree_random.pdf') 
visualizeTree(initRoot2, graph2)
graph2.render(view=True)

# Evaluate on test set
a2 = successRate(testdf, initRoot2)

print('\n\nRANDOM:\nSuccess rate: ',a2,'%')


    


        
    
  
    
