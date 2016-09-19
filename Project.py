
# coding: utf-8

# In[4]:

import pandas as pd
import json

data = pd.read_csv('data.csv',header=-1)
data = data.drop(data.columns[330], 1)
data.head()
#First 20 columns are cdf values (inputs). 20 for p1 then 20 for p2.
#Next 310 columns are y values (outputs). For card 1, bet 0,0.1,...,3, then card 2 ... (10 cards, 30 sizes per card)


# In[5]:

#first 20 columns are X inputs. cdf for P1 (10 cards), then cdfs for P2 (10 cards).
data_X = data[data.columns[0:20]]
data_X.head()


# In[6]:

#next 310 columns are y outputs. probability of betting size 0,...,3 with 1, 0,...,3 with 2, ..., 10
data_y = data[data.columns[20:]]
data_y.head()


# In[7]:

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, 
                                                    test_size=0.2, random_state=7)


# In[8]:

X_train.head()


# In[3]:

#some constants
num_cards = 10
num_bets = 31
max_y_distance = num_bets - 1
max_x_distance = num_cards - 1
num_inputs = 20
start_index = num_inputs
num_players = 2


# In[4]:

#custom distance function, modified version of earth mover's distance, averaged over the 10 cards
def emd_y(array1, array2):
    result_total = 0
    for i in range (num_cards):
        start = start_index + i * num_bets
        end = start + num_bets
        result = 0
        delta = 0
        for j in range(start, end):
            delta = delta + array1[j] - array2[j]
            result = result + abs(delta)
        result = result / max_y_distance
        result_total = result_total + result
    result_total = result_total / num_cards
    return result_total


# In[5]:

#Same as emd_y but assumes input is dataframe instead of array
def emd_df_y(df1, df2):
    result_total = 0
    for i in range (num_cards):
        start = start_index + i * num_bets
        end = start + num_bets
        result = 0
        delta = 0
        for j in range(start, end):
            delta = delta + df1.loc[j] - df2.loc[j]
            result = result + abs(delta)
        result = result / max_y_distance
        result_total = result_total + result
    result_total = result_total / num_cards
    return result_total


# In[6]:

#convert probability cdf into pdf before applying the x distance function
#Still use the cdf as features, but pdf is better for using earth mover's distance.
def cdf_to_pdf(cdf):
    pdf = range(num_inputs)
    pdf[0] = cdf[0]
    for i in range(1,num_cards):
        pdf[i] = cdf[i] - cdf[i-1]
    pdf[num_cards] = cdf[num_cards]
    for i in range(num_cards+1,num_inputs):
        pdf[i] = cdf[i] - cdf[i-1]
    return pdf        


# In[7]:

#custom distance function, modified version of earth mover's distance, averaged over the 2 players
def emd_x(cdf1, cdf2):
    #first convert from cdfs to pdfs, so we can compute emd.
    pdf1 = cdf_to_pdf(cdf1)
    pdf2 = cdf_to_pdf(cdf2)
    result_total = 0
    #compute emd separately for each player, then average them
    for i in range (num_players):
        start = i * num_cards
        end = start + num_cards
        #this part is standard EMD "Hungarian" algorithm
        result = 0
        delta = 0
        for j in range(start, end):
            delta = delta + pdf1[j] - pdf2[j]
            result = result + abs(delta)
        # normalize it so max possible distance is 1 (i.e., if moved all mass from leftmost column to rightmost column) 
        result = result / max_x_distance; 
        result_total = result_total + result
    # average over the emd of both players
    result_total = result_total / num_players
    return result_total


# In[8]:

#same as emd_x but assumes dataframe inputs instead of arrays
def emd_df_x(cdf1, cdf2):
    pdf1 = cdf_to_pdf(cdf1)
    pdf2 = cdf_to_pdf(cdf2)
    result_total = 0
    for i in range (num_players):
        start = i * num_cards
        end = start + num_cards
        result = 0
        delta = 0
        for j in range(start, end):
            delta = delta + pdf1[j] - pdf2[j]
            result = result + abs(delta)
        result = result / max_x_distance;
        result_total = result_total + result
    result_total = result_total / num_players
    return result_total


# In[70]:

x1 = data_X.ix[0]
x2 = data_X.ix[3]
distance = emd_df_x(x1,x2)
print (distance)


# In[71]:

#sanity check test
y1 = data_y.ix[0]
y2 = data_y.ix[3]
distance = emd_y(y1, y2)
print (distance)


# In[ ]:




# In[59]:

#sanity check test
y1 = data_y.ix[0]
y2 = data_y.ix[3]
distance = emd_df_y(y1, y2)
print (distance)


# In[84]:

#Need to overwrite knn prediction function to deal with format of y (predicting multiple output probabilities not just one).
#This needs to be fixed. I started with template from lecture and am modifying it.

def knn_predict(test_example, train_data, k):
    """Make a classification prediction using kNN
    
    
    Parameters
    ----------
    test_example (array or list-like): the new observation we need to predict
    train_data (dataframe or dict-like): the training data, assuming the labels
    are the last column
    k (int): the number of neighbors
    
    Returns the predicted label for test_example based
    on Euclidean distance and uniform weighting
    
    Returns
    ----------
    The predicted label for test_example based on Euclidean distance and
    uniform weighting
    """
    # find the squared difference btwn
    # test example and each row in training data
    #sq_diff = (test_example - train_data.iloc[:, :-1])**2
    
    dist = range(len(train_data))
    
    #use custom emd distance function
    for i in range(len(train_data)):
        dist[i] = emd_df_x(train_data.iloc[i],test_example)
    
    # find total distance for each row by summing 
    # across the row axis
    # YOUR CODE IN 1 LINE BELOW:
    #dist = np.sum(sq_diff, axis=1)
    
    # append the labels column back so we can use them
    # YOUR CODE IN 1 LINE BELOW:
    #dist = pd.concat([dist, train_data.iloc[:, -1]], axis=1)
    
    
    
    # rename the columns of our dataframe
    # the distance column should be called 'distance'
    # and the labels column should be called 'labels'
    # YOUR CODE IN 1 LINE BELOW:
    #dist.columns = ['distance', 'labels']
    
    
    
    # sort by distance to find closest neighbors
    # and only keep the k closest neighbors
    # YOUR CODE BELOW:
    dist = dist.sort_values(by='distance', ascending=True).head(k)
    
    # if nothing has 2+ occurrences, mode() will return an empty
    # Series, so in this case we just pick the 1st result
    if dist['labels'].mode().size == 0:
        return dist['labels'].iloc[0]
    else:
        # return the most frequently occuring label as the prediction
        # breaks ties by picking the first mode
        return dist['labels'].mode().iloc[0]


# In[50]:

type(data_y)


# In[41]:

from sklearn.neighbors import KNeighborsClassifier
#knn = KNeighborsClassifier()

knn = KNeighborsClassifier(n_neighbors=4, algorithm='ball_tree',
          metric=emd_df_x)
#knn = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', p=2)
#knn = KNeighborsClassifier(n_neighbors=1)
#knn =  knn_predict(data_y.ix[0], data_X, 3)
# starting with k=1
# this creates the classifier
#knn = KNeighborsClassifier(n_neighbors=1, weights='uniform', p=2)


# In[44]:

#type(X_train)
#type(X_test[0])
type(data_X[0])


# In[81]:

emd_x(X_train.iloc[4],X_test.iloc[0])


# In[91]:

#runs into some issues because knn_predict function not correct
#knn_predict(X_test.iloc[0], X_train, 2)


# In[86]:

#knn.fit(data_X, data_y)
# use apply to generate predictions
#pd.DataFrame(X_test.apply(lambda x: knn_predict(x, X_train, 2), axis=1)).rename(columns={0: 'Output'})


# In[29]:

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()


# In[92]:

# Fits the model
#dtc.fit(data_X, data_y)

#dtc.score(data_X, data_y)


# In[ ]:



