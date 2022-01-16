# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
import os
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation
from sklearn.metrics.pairwise import pairwise_distances
import ipywidgets as widgets
from IPython.display import display, clear_output
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')


# %%
# read books csv file
books = pd.read_csv('BX-Books.csv', sep=';',
                    error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication',
                 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
# read users csv file
users = pd.read_csv('BX-Users.csv', sep=';',
                    error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']
# read ratings csv file
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';',
                      error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']


# %%
# drop unimportant feilds
# books.drop(['imageUrlS','imageUrlM','imageUrlL'],axis = 1, inplace = True)


# %%
books.dtypes


# %%
# making this setting to display full text in columns
pd.set_option('display.max_colwidth', -1)


# %%
# checking yearOfPublication in books
books.yearOfPublication.unique()


# %%
# two feilds are incorrect
books.loc[books.yearOfPublication == 'DK Publishing Inc', :]


# %%
# making edits #1
books.loc[books.ISBN == '078946697X',
          'bookTitle'] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"
books.loc[books.ISBN == '078946697X', 'bookAuthor'] = "Michael Teitelbaum"
books.loc[books.ISBN == '078946697X', 'yearOfPublication'] = 2000
books.loc[books.ISBN == '078946697X', 'publisher'] = "DK Publishing Inc"

# making edits #2
books.loc[books.ISBN == '0789466953',
          'bookTitle'] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"
books.loc[books.ISBN == '0789466953', 'bookAuthor'] = "James Buckley"
books.loc[books.ISBN == '0789466953', 'yearOfPublication'] = 2000
books.loc[books.ISBN == '0789466953', 'publisher'] = "DK Publishing Inc"

# output
books.loc[(books.ISBN == '078946697X') | (books.ISBN == '0789466953'), :]


# %%
books.loc[books.yearOfPublication == 'Gallimard', :]


# %%
# making other edits
books.loc[books.ISBN == '2070426769',
          'bookTitle'] = "Peuple du ciel, suivi de 'Les Bergers"
books.loc[books.ISBN == '2070426769',
          'bookAuthor'] = "Jean-Marie Gustave Le ClÃ?Â©zio"
books.loc[books.ISBN == '2070426769', 'yearOfPublication'] = 2003
books.loc[books.ISBN == '2070426769', 'publisher'] = "Gallimard"

# output
books.loc[books.ISBN == '2070426769', :]


# %%
# changing data types from object to numeric
books.yearOfPublication = pd.to_numeric(
    books.yearOfPublication, errors='coerce')

# output
books.yearOfPublication.dtypes


# %%
# checking each unique value of yearOfPublication for errors
sorted(books['yearOfPublication'].unique())


# %%
# changing values with NAN
books.loc[(books.yearOfPublication > 2006) | (
    books.yearOfPublication == 0), 'yearOfPublication'] = np.NAN


# %%
# replacing with mean of years
books.yearOfPublication.fillna(
    round(books.yearOfPublication.mean()), inplace=True)
books.yearOfPublication.dtypes

# changing dtype from float64 to int64
books.yearOfPublication = books.yearOfPublication.astype(np.int64)

# output
books.yearOfPublication.dtypes

# here yearOfPublication column is fully cleaned


# %%
# checking publisher in books
books.loc[books.publisher.isnull(), :]


# %%
# editing feilds
books.loc[(books.ISBN == '193169656X'), 'publisher'] = 'other'
books.loc[(books.ISBN == '1931696993'), 'publisher'] = 'other'

# output
books.loc[(books.ISBN == '193169656X') | (books.ISBN == '1931696993'), :]

# here books are fully cleaned


# %%
# now cheking users table
users.dtypes


# %%
# checking each value of age in users data
sorted(users.Age.unique())


# %%
# replacing unimportant data with nan and then editing it with mean
users.loc[(users.Age > 90) | (users.Age < 5), 'Age'] = np.nan
users.Age = users.Age.fillna(users.Age.mean())

# changing data type from float64 to int32
users.Age = users.Age.astype(np.int32)
users.Age.dtypes

# here users data is fully cleaned


# %%
# now checking rating dataset
ratings.dtypes

# ratings dataset is OK


# %%
books.dtypes


# %%
# dict = {'ISBN':books.ISBN,
#         'bookTitle':books.bookTitle,
#         'bookAuthor':books.bookAuthor,
#         'yearOfPublication':books.yearOfPublication,
#         'publisher':books.publisher,
#         'imageUrlS':books.imageUrlS,
#         'imageUrlM':books.imageUrlM,
#         'imageUrlL':books.imageUrlL
# }
# pd.DataFrame(dict).to_csv(r'C:/Users/ShubhamJain/Documents/python/data-science/book-recommender-medium/cleaned/books.csv')
# new_books = pd.read_csv(r'C:/Users/ShubhamJain/Documents/python/data-science/book-recommender-medium/cleaned/books.csv',sep=";", error_bad_lines=False)


# %%
# new_books.head(4)


# %%
users.head(5)


# %%
# From here anylysis starts


# %%
ratings.shape


# %%
n_users = users.shape[0]  # number of users
n_books = books.shape[0]  # number of books
print(n_users*n_books)


# %%
ratings.dtypes


# %%
# ratings should have only those books which are in the books dataset
ratings_new = ratings[ratings.ISBN.isin(books.ISBN)]


# %%
print(ratings.shape)
print(ratings_new.shape)
print(ratings.shape[0]-ratings_new.shape[0])


# %%
print("number of users in users dataset: " + str(n_users))
print("number of books in books dataset: " + str(n_books))


# %%
sparsity = 1.0 - len(ratings_new) / float(n_users*n_books)


# %%
print(sparsity*100)


# %%
# segragating implicit and explicit ratings
ratings_explicit = ratings_new[ratings_new.bookRating != 0]
ratings_implicit = ratings_new[ratings_new.bookRating == 0]


# %%
print(ratings_new.shape)
print(ratings_explicit.shape)
print(ratings_implicit.shape)


# %%
sns.countplot(data=ratings_explicit, x='bookRating')
plt.show()


# %%
# popularity based recommendation system


# %%
ratings_count = pd.DataFrame(
    ratings_explicit.groupby(['ISBN'])['bookRating'].sum())
top10 = ratings_count.sort_values('bookRating', ascending=False).head(10)
top10.merge(books, left_index=True, right_on='ISBN')


# %%
users_exp_ratings = users[users.userID.isin(ratings_explicit.userID)]
users_imp_ratings = users[users.userID.isin(ratings_implicit.userID)]
print(users.shape)
print(users_exp_ratings.shape)
print(users_imp_ratings.shape)


# %%
# collabrative filtering based recommendation system


# %%
counts1 = ratings_explicit['userID'].value_counts()
ratings_explicit = ratings_explicit[ratings_explicit['userID'].isin(
    counts1[counts1 >= 100].index)]
counts = ratings_explicit['bookRating'].value_counts()
ratings_explicit = ratings_explicit[ratings_explicit['bookRating'].isin(
    counts[counts >= 100].index)]


# %%
# Generating ratings matrix from explicit ratings table
ratings_matrix = ratings_explicit.pivot(
    index='userID', columns='ISBN', values='bookRating')
userID = ratings_matrix.index
ISBN = ratings_matrix.columns
print(ratings_matrix.shape)
ratings_matrix.head()
# Notice that most of the values are NaN (undefined) implying absence of ratings


# %%
# considering only those users who gave explicit ratings
n_users = ratings_matrix.shape[0]
n_books = ratings_matrix.shape[1]
print(n_users, n_books)


# %%
# since NaNs cannot be handled by training algorithms, replacing these by 0, which indicates absence of ratings
# setting data type
ratings_matrix.fillna(0, inplace=True)
ratings_matrix = ratings_matrix.astype(np.int32)


# %%
# checking first few rows
ratings_matrix.head(5)


# %%
# rechecking the sparsity
sparsity = 1.0-len(ratings_explicit)/float(users_exp_ratings.shape[0]*n_books)
print('The sparsity level of Book Crossing dataset is ' + str(sparsity*100) + ' %')


# %%
# training our recommendation model


# %%
# setting global variables
global metric, k
k = 10
metric = 'cosine'


# %%
# user-based recomendation sysytem


# %%
# This function finds k similar users given the user_id and ratings matrix
# These similarities are same as obtained via using pairwise_distances
def findksimilarusers(user_id, ratings, metric=metric, k=k):
    similarities = []
    indices = []
    model_knn = NearestNeighbors(metric=metric, algorithm='brute')
    model_knn.fit(ratings)
    loc = ratings.index.get_loc(user_id)
    distances, indices = model_knn.kneighbors(
        ratings.iloc[loc, :].values.reshape(1, -1), n_neighbors=k+1)
    similarities = 1-distances.flatten()

    return similarities, indices


# %%
# This function predicts rating for specified user-item combination based on user-based approach
def predict_userbased(user_id, item_id, ratings, metric=metric, k=k):
    prediction = 0
    user_loc = ratings.index.get_loc(user_id)
    item_loc = ratings.columns.get_loc(item_id)
    # similar users based on cosine similarity
    similarities, indices = findksimilarusers(user_id, ratings, metric, k)
    # to adjust for zero based indexing
    mean_rating = ratings.iloc[user_loc, :].mean()
    sum_wt = np.sum(similarities)-1
    product = 1
    wtd_sum = 0

    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i] == user_loc:
            continue
        else:
            ratings_diff = ratings.iloc[indices.flatten(
            )[i], item_loc] - np.mean(ratings.iloc[indices.flatten()[i], :])
            product = ratings_diff * (similarities[i])
            wtd_sum = wtd_sum + product

    # in case of very sparse datasets, using correlation metric for collaborative based approach may give negative ratings
    # which are handled here as below
    if prediction <= 0:
        prediction = 1
    elif prediction > 10:
        prediction = 10

    prediction = int(round(mean_rating + (wtd_sum/sum_wt)))
    print(
        '\nPredicted rating for user {0} -> item {1}: {2}'.format(user_id, item_id, prediction))

    return prediction


# %%
predict_userbased(11676, '0001056107', ratings_matrix)


# %%
# item based recommendation system


# %%
# This function finds k similar items given the item_id and ratings matrix

def findksimilaritems(item_id, ratings, metric=metric, k=k):
    similarities = []
    indices = []
    ratings = ratings.T
    loc = ratings.index.get_loc(item_id)
    model_knn = NearestNeighbors(metric=metric, algorithm='brute')
    model_knn.fit(ratings)

    distances, indices = model_knn.kneighbors(
        ratings.iloc[loc, :].values.reshape(1, -1), n_neighbors=k+1)
    similarities = 1-distances.flatten()

    return similarities, indices


# %%
similarities, indices = findksimilaritems('0001056107', ratings_matrix)


# %%
# This function predicts the rating for specified user-item combination based on item-based approach
def predict_itembased(user_id, item_id, ratings, metric=metric, k=k):
    prediction = wtd_sum = 0
    user_loc = ratings.index.get_loc(user_id)
    item_loc = ratings.columns.get_loc(item_id)
    # similar users based on correlation coefficients
    similarities, indices = findksimilaritems(item_id, ratings)
    sum_wt = np.sum(similarities)-1
    product = 1
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i] == item_loc:
            continue
        else:
            product = ratings.iloc[user_loc,
                                   indices.flatten()[i]] * (similarities[i])
            wtd_sum = wtd_sum + product
    prediction = int(round(wtd_sum/sum_wt))

    # in case of very sparse datasets, using correlation metric for collaborative based approach may give negative ratings
    # which are handled here as below //code has been validated without the code snippet below, below snippet is to avoid negative
    # predictions which might arise in case of very sparse datasets when using correlation metric
    if prediction <= 0:
        prediction = 1
    elif prediction > 10:
        prediction = 10

    print(
        '\nPredicted rating for user {0} -> item {1}: {2}'.format(user_id, item_id, prediction))

    return prediction


# %%
prediction = predict_itembased(11676, '0001056107', ratings_matrix)


# %%
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# %%
# This function utilizes above functions to recommend items for item/user based approach and cosine/correlation.
# Recommendations are made if the predicted rating for an item is >= to 6,and the items have not been rated already
def recommendItem(user_id, ratings, metric=metric):
    if (user_id not in ratings.index.values) or type(user_id) is not int:
        print("User id should be a valid integer from this list :\n\n {} ".format(
            re.sub('[\[\]]', '', np.array_str(ratings_matrix.index.values))))
    else:
        ids = ['Item-based (correlation)', 'Item-based (cosine)',
               'User-based (correlation)', 'User-based (cosine)']
        select = widgets.Dropdown(
            options=ids, value=ids[0], description='Select approach', width='1000px')

        def on_change(change):
            clear_output(wait=True)
            prediction = []
            if change['type'] == 'change' and change['name'] == 'value':
                if (select.value == 'Item-based (correlation)') | (select.value == 'User-based (correlation)'):
                    metric = 'correlation'
                else:
                    metric = 'cosine'
                with suppress_stdout():
                    if (select.value == 'Item-based (correlation)') | (select.value == 'Item-based (cosine)'):
                        for i in range(ratings.shape[1]):
                            # not rated already
                            if (ratings[str(ratings.columns[i])][user_id] != 0):
                                prediction.append(predict_itembased(
                                    user_id, str(ratings.columns[i]), ratings, metric))
                            else:
                                # for already rated items
                                prediction.append(-1)
                    else:
                        for i in range(ratings.shape[1]):
                            # not rated already
                            if (ratings[str(ratings.columns[i])][user_id] != 0):
                                prediction.append(predict_userbased(
                                    user_id, str(ratings.columns[i]), ratings, metric))
                            else:
                                # for already rated items
                                prediction.append(-1)
                prediction = pd.Series(prediction)
                prediction = prediction.sort_values(ascending=False)
                recommended = prediction[:10]
                print("As per {0} approach....Following books are recommended...".format(
                    select.value))
                for i in range(len(recommended)):
                    print("{0}. {1}".format(
                        i+1, books.bookTitle[recommended.index[i]].encode('utf-8')))
        select.observe(on_change)
        display(select)


# %%
# checking for incorrect entries
recommendItem(999999, ratings_matrix)


# %%
recommendItem(4385, ratings_matrix)


# %%
recommendItem(4385, ratings_matrix)


# %%

