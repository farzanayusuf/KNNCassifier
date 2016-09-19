import pandas as pd
import json

data1 = pd.read_csv('data.csv',header=-1)
data1 = data1.drop(data1.columns[330], 1)

#some constants
num_cards = 10
num_bets = 31
max_y_distance = num_bets - 1
max_x_distance = num_cards - 1
num_inputs = 20
start_index = num_inputs
num_players = 2

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
import array
#convert probability cdf into pdf before applying the x distance function
#Still use the cdf as features, but pdf is better for using earth mover's distance.
def cdf_to_pdf(cdf):
    pdf = list(range(num_inputs))
    #print (range(num_inputs))
    #pdf = None
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
    #print (result_total)
    return result_total


# Need to overwrite knn prediction function to deal with format of y (predicting multiple output probabilities not just one).
# This needs to be fixed. I started with template from lecture and am modifying it.
from scipy.stats import mode

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
    # sq_diff = (test_example - train_data.iloc[:, :-1])**2

    dist_old = list(range(len(train_data)))

    # use custom emd distance function
    for i in range(len(train_data)):
        dist_old[i] = emd_df_x(train_data.iloc[i], test_example)

    # find total distance for each row by summing
    # across the row axis
    # YOUR CODE IN 1 LINE BELOW:
    # dist = np.sum(sq_diff, axis=1)

    dist = pd.DataFrame(dist_old)

    # append the labels column back so we can use them
    # YOUR CODE IN 1 LINE BELOW:
    #print ( train_data.iloc[:])
    dist = pd.concat([dist, train_data.iloc[:,-1]], axis=1)
    #print (dist)

    # rename the columns of our dataframe
    # the distance column should be called 'distance'
    # and the labels column should be called 'labels'
    # YOUR CODE IN 1 LINE BELOW:
    dist.columns = ['distance', 'labels']


    # sort by distance to find closest neighbors
    # and only keep the k closest neighbors
    # YOUR CODE BELOW:
    dist = dist.sort_values(by='distance', ascending=True).head(k)

    return (dist.index.tolist()[0])


from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor


knn = KNeighborsClassifier(n_neighbors=1, weights='uniform', p=4)

import csv
percentage_inc= .01
percentage =200

data = data1.ix[0:percentage]

with open('result_with_diffrent_size.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ',
                            lineterminator='\n',
                           quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range (1,20):
        data = data1.ix[0:500]

        #first 20 columns are X inputs. cdf for P1 (10 cards), then cdfs for P2 (10 cards).
        data_X = data[data.columns[0:20]]

        #next 310 columns are y outputs. probability of betting size 0,...,3 with 1, 0,...,3 with 2, ..., 10
        data_y = data[data.columns[20:]]
        data_y.head()

        from sklearn.cross_validation import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(data_X, data_y,
                                                            test_size=.05*i, random_state=7)


        distance_total=0
        length = len(X_test.index)
        knn = KNeighborsRegressor(n_neighbors=4, algorithm='ball_tree',metric=emd_df_x)
        # knn.fit(X_train,y_train)
        # knn.predict(X_test)
        # knn.score(X_test,y_test)
        # result_list=[]

        for index in range(length) :
            result_list = []
            x= (knn_predict(X_test.iloc[index], X_train, 4))
            y = emd_df_y(y_test.iloc[index], y_train.iloc[x])
            distance_total +=  y
            print(y)
            index+= 1


        distance_total= distance_total/length
        result_list.append(distance_total)
        result_list.append(knn.score(X_test, y_test))
        spamwriter.writerow(result_list)

        print ("Average:"+ str( distance_total))
