#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import json


# In[7]:


# Read file and make a pandas dataframe object of it.

myFile = pd.read_csv('550-p1-cset-krk-1.csv')
myFile
myFrame=pd.DataFrame(myFile.values, columns = ["White-King-File", "White-King-rank", "White-Rook-file", "White-Rook-rank","Black-King-file","Black-King-rank","Class"])


# In[8]:


# Split into training_set:60%, holdout_set:20%, validation_set: 20%

training_set = myFrame.sample(frac=0.6)
remaining_set = myFrame.drop(training_set.index)
holdout_set = remaining_set.sample(frac=0.5)
validation_set = remaining_set.drop(holdout_set.index)

training_set.reset_index(drop=True, inplace=True)
holdout_set.reset_index(drop=True, inplace=True)
validation_set.reset_index(drop=True, inplace=True)


# In[9]:


# Entropy calculation equation

def calculate_entropy(label):
    class_list,class_count = np.unique(label,return_counts = True)
    entropy = np.sum([(-class_count[i]/np.sum(class_count))*np.log2(class_count[i]/np.sum(class_count)) 
                        for i in range(len(class_list))])
    return entropy


# In[10]:


# dataset = training_set
# attribute = attribute of the dataset, i.e column name (training_set.columns[:-1]) E.g White-King-File
# label = Name of Class label colums, i.e. 'Class'
# Information gain calculation Function
def calculate_information_gain(dataframe,attribute,label): 
    dataframe_entropy = calculate_entropy(dataframe[label])   
    values, feat_counts= np.unique(dataframe[attribute],return_counts=True)
    
                              
    weighted_feature_entropy = np.sum([(feat_counts[i]/np.sum(feat_counts))*calculate_entropy(dataframe.where(dataframe[attribute]
                              ==values[i]).dropna()[label]) for i in range(len(values))])    
    information_gain = dataframe_entropy - weighted_feature_entropy    
    return information_gain


# In[11]:


# Decision Node class which is equivalent to QNode

class DNode:
    def __init__(self, attribute, values, info_gain):
        self.attribute = attribute
        self.values = values
        self.info_gain = info_gain
        self.children = {}    


# In[12]:


# Leaf node class

class LeafNode:
    def __init__(self,value):
        self.value = value
        


# In[13]:


# Function to return a copy of list

def copy_list(l):
    return l[:]


# In[14]:


# This function passes dataset, attributes and class_label='Class'
# Returns a DNode or a LeafNode object based on the information gain value.

def select_attribute(dataset, attributes, class_label):
    if len(attributes)<=0:
        return LeafNode(np.unique(dataset[class_label])[0])
    
    gains = [calculate_information_gain(dataset, attribute, class_label) for attribute in attributes]
#     print("gains",gains)
    maxGainIndex = np.argmax(gains)
    maxGainValue = gains[maxGainIndex]
#     print("maxGainIndex",maxGainIndex)
#     print("maxGainValue",maxGainValue)
#     print("highGainAttribute:",attributes[maxGainIndex])        
    
    if maxGainValue == 0.0:
        value = np.unique(dataset[class_label])[0]
#         print('value',value)
        return LeafNode(value)
    
    else:
        newAttribute = attributes[maxGainIndex]
        values = np.unique(dataset[newAttribute])
#         print("newAttribute:",newAttribute)
#         print("values:",values)
        return DNode(newAttribute, values, maxGainValue)
        


# In[15]:


# This Function recursively creates a Decision tree by using select_attribute Function on each child nodes.

def create_dtree(dataset, attributes, class_label):            
    node = select_attribute(dataset, attributes, class_label)
    
    if isinstance(node, DNode):                        
        newAttributes = copy_list(attributes)
        newAttributes.remove(node.attribute)        
        for v in node.values:
            newDataSet = dataset.where(dataset[node.attribute] == v).dropna()
            if(newDataSet.shape[0]==0):
                continue
            node.children[v] = create_dtree(newDataSet, newAttributes, class_label)
    
    return node


# In[16]:


attributes = training_set.columns[:-1].tolist()
class_label = 'Class'
rootNode = create_dtree(training_set, copy_list(attributes), class_label)


# In[17]:


# Function to traverse through the tree

def check_treeRec(node):

    if isinstance(node, LeafNode):        
        print('\tleaf node value:', node.value)
        return 
    print("Parent: ",node.attribute)
    for k,v in node.children.items():
        if isinstance(v, DNode):
            print("\tChild: ",v.attribute)
    
    for k,v in node.children.items():                
        check_treeRec(v)
    
    return 


# In[18]:


def display_information_gain(node):
    if isinstance(node, LeafNode):        
        return 
    print("Decision Node: ",node.attribute, ", Info_gain: ", node.info_gain)        
    for k,v in node.children.items():                
        display_information_gain(v)    
    return 


# In[19]:


# Function to predict whether the prediction for the input matches with the correct answer.

def predict(node, row, correctAns):
    if isinstance(node, LeafNode):                
        return node.value == correctAns
#     print('Attr:',node.attribute,", value:", row[node.attribute])
    if row[node.attribute] not in node.children:
        return False
    return predict(node.children[row[node.attribute]], row, correctAns)
    


# In[20]:


# Function that takes a dataset as an input and runs the predict() function for all rows and calculates the accuracy of
# decision tree classifier. Returns misclassed_rows dataframe.

def calculate_prediction(rootNode, dataset):
    count = 0
    miscount = 0
    misclassed_rows = pd.DataFrame(columns = ["White-King-File", "White-King-rank", "White-Rook-file", "White-Rook-rank","Black-King-file","Black-King-rank","Class"])
    for i in range(dataset.shape[0]):
        ans = predict(rootNode, dataset.loc[i], dataset.loc[i][-1])
        if ans:
            count+=1
        else:
            miscount+=1
            misclassed_rows = misclassed_rows.append(dataset.loc[i], ignore_index = True)            
    accuracy = (count*100)/dataset.shape[0]
    print("miscount:",miscount,"count:",count,", accuracy = ", accuracy,"%")
    return (misclassed_rows, accuracy)


# In[21]:


misclassed_set, accuracy_tree1 = calculate_prediction(rootNode, holdout_set)


# In[22]:


miss_repeated = pd.concat([misclassed_set]*3, ignore_index=True)


# In[23]:


collected_set = miss_repeated.append(training_set, ignore_index = True)


# In[24]:


training_set2 = collected_set.sample(training_set.shape[0], replace=True)


# In[25]:


holdout_set2 = collected_set.drop(index = training_set2.index)
holdout_set2.reset_index(drop=True, inplace=True)


# In[26]:


training_set2.reset_index(drop=True, inplace=True)


# In[27]:


dtree2_node = create_dtree(training_set2, copy_list(attributes), class_label)


# In[28]:


misclassed_set2, accuracy_tree2 = calculate_prediction(dtree2_node, holdout_set2)


# In[29]:


# It is the ensemble classifier. Builder takes two tree root nodes (classifiers), dataset, attributes and class label. 
# It initializes weights of both classifiers to 1 and for each misclassification by each classifier, it halves the weights.
# It uses predict function and the classification of the classifier with higher weight is considered.
# Returns the weights of both classifiers and calculates the accuracy of the ensemble classifier

def builder(tree1, tree2, dataset, attributes, class_label):
    w1 = 1
    w2 = 1
    miscount=0
    for i in range(dataset.shape[0]):
        ans1 = predict(tree1, dataset.loc[i], dataset.loc[i][-1])
        ans2 = predict(tree2, dataset.loc[i], dataset.loc[i][-1])
        
        ans = False
        if(not ans1):
            w1 = w1/2
        if(not ans2):
            w2 = w2/2
        if(w1>=w2):
            ans = ans1
        else:
            ans = ans2
        if not ans:
            miscount+=1
#         print("ans1:",ans1,", ans2:",ans2,", ans:",ans,", w1:",w1,", w2:",w2,", miscount:",miscount)
    accuracy = 100-((miscount*100)/dataset.shape[0])
    print("miscount:",miscount,", accuracy = ", accuracy,"%")
    weights = [w1,w2]
    return (weights, accuracy)


# In[30]:


weights, ensemble_accuracy = builder(rootNode, dtree2_node, validation_set, copy_list(attributes), class_label)


# In[31]:


validation_set_tree1_accuracy = calculate_prediction(rootNode, validation_set)[1]
validation_set_tree2_accuracy = calculate_prediction(dtree2_node, validation_set)[1]


# In[32]:


print("\nTraining_set for Tree 1 of size: ",training_set.shape)
print(training_set)
print("-------------------------------------------------------------------------------------------------------------")
print("\nHoldout_set for Tree 1 of size: ",holdout_set.shape)
print(holdout_set)
print("-------------------------------------------------------------------------------------------------------------")
print("\nTraining_set for Tree 2 of size: ",training_set2.shape)
print(training_set2)
print("-------------------------------------------------------------------------------------------------------------")
print("\nHoldout_set for Tree 2 of size: ",holdout_set2.shape)
print(holdout_set2)
print("-------------------------------------------------------------------------------------------------------------")
print("\nValidation set of size: ",validation_set.shape)
print(validation_set)


# In[174]:


print("\n\nMisclassed holdout of size: ", misclassed_set.shape)
print(misclassed_set)


# In[175]:


print("Accuracy of Tree 1 on Holdout set 1:", accuracy_tree1, "%")
print("Accuracy of Tree 2 on Holdout set 2:", accuracy_tree2, "%")
print("Accuracy of Tree 1 on Validation set:", validation_set_tree1_accuracy, "%")
print("Accuracy of Tree 2 on Validation set:", validation_set_tree2_accuracy, "%")

print("Accuracy of Tree 2:", ensemble_accuracy, "%")


# In[176]:


print("Ensemble Voting weights : w1=",weights[0],", w2=",weights[1])


# In[177]:


print("Parent child node tree display and class for each leaf node, for decision TREE 1:\n")
check_treeRec(rootNode)


# In[178]:


print("Parent child node tree display and class for each leaf node, for decision TREE 2:\n")
check_treeRec(dtree2_node)


# In[186]:


print("Information gain for each Decision node for Decision tree 1:")
display_information_gain(rootNode)


# In[ ]:




