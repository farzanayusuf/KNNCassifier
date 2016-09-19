import pandas as pd
import json

data1 = pd.read_csv('data.csv',header=-1)
data1 = data1.drop(data1.columns[330], 1)

data = data1.ix[0:5000]
data.head()
#First 20 columns are cdf values (inputs). 20 for p1 then 20 for p2.
#Next 310 columns are y values (outputs). For card 1, bet 0,0.1,...,3, then card 2 ... (10 cards, 30 sizes per card)

# In[5]:

#first 20 columns are X inputs. cdf for P1 (10 cards), then cdfs for P2 (10 cards).
data_X = data[data.columns[0:20]]
#print (data_X.head(2))

# In[6]:

#next 310 columns are y outputs. probability of betting size 0,...,3 with 1, 0,...,3 with 2, ..., 10
data_y = data[data.columns[20:]]
data_y.head()
#print (data_y.head())

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y,
                                                    test_size=0.2, random_state=7)


# In[8]:

#print (X_train.head(1))


# In[3]:

#some constants
num_cards = 10
num_bets = 31
max_y_distance = num_bets - 1
max_x_distance = num_cards - 1
num_inputs = 20
start_index = num_inputs
num_players = 2

from scipy.stats import mode
from collections import Counter
import csv
import math
def TD(p,node) :

  if p is None : return
  q=[]
  left = p.tree_.children_left
  right = p.tree_.children_right
  threshold = p.tree_.threshold
  # error = tree.tree_.init_error

  value = p.tree_.value
  print ("hey")
  print(q)

  def printtree(left, right, threshold, node):
      if (left[node] == -1 and right[node] == -1):
          print ("returning")
          return;
      if (threshold[node] != -2):
          if left[node] != -1:
              print(str(threshold[node]))
              printtree(left, right, threshold,  left[node])
          if right[node] != -1:
              printtree(left, right, threshold , right[node])


  printtree(left,right,threshold,0)
def get_code(tree, feature_names, offset_unit='    '):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    #error = tree.tree_.init_error
    features = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    with open("Output1.txt", "w") as text_file:
        def recurse(left, right, threshold, features, node, depth=0):
            offset = offset_unit * depth
            if (threshold[node] != -2):
                text_file.writelines(offset + "if ( " + str(features[node]) + " <= " + str(threshold[node]) + " ) {")
                if left[node] != -1:
                    recurse(left, right, threshold, features, left[node], depth + 1)
                text_file.writelines(offset + "} else {")
                if right[node] != -1:
                    recurse(left, right, threshold, features, right[node], depth + 1)
                text_file.writelines(offset + "}")
            else:
                text_file.writelines(offset + "return ")

        recurse(left, right, threshold, features, 0, 0)


type(data_y)



from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn import tree

dtc = tree.DecisionTreeRegressor(criterion="mse",max_depth=10)


# In[92]:

# Fits the model
dtc.fit(X_train, y_train)
y1=dtc.predict(X_test.iloc[0])
y= pd.DataFrame()
#y.iloc[0]=y_train.iloc[0]
from sklearn.externals.six import StringIO
with open("result.dot", 'w') as f:
  f = tree.export_graphviz(dtc, out_file=f,class_names =False, filled=False, label =None)
import os
from inspect import getmembers
#print( getmembers( dtc.tree_ ) )
from sklearn.externals.six import StringIO
import pydotplus
from os import system
import matplotlib.pyplot as plt

dot_data = StringIO()
tree.export_graphviz(dtc, out_file=dot_data,
                     class_names=False, filled=False, label=None,rotate= True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#Image(graph.create_png())
# graph.write_pdf("iris.pdf")
plt.figure()
#system("dot -Tpng G:/FIU/Fall 16/Research/result.dot -o G:/FIU/Fall 16/Research/dtree2.png")

#
# #plt.scatter(X, y, c="k", label="data")
# plt.plot(X_test, y, c="g", label="max_depth=2", linewidth=2)
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Decision Tree Regression")
# plt.legend()
# plt.show()
print(dtc.score(X_test, y_test))
#pd.concat(y,y1)
#print(emd_df_y(y_test.iloc[0],y))
#get_code(dtc, y_train.columns)
#print (dtc.max_depth)
#print (TD(dtc,0))

