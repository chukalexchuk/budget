import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import pandas as pd
import matplotlib.pyplot as plt
import spacy_sentence_bert

df = pd.read_csv("data/data.csv")  # use your data to train the model
df_simple = df.loc[:, ("Code", "Amount")]

df_simple["Category"] = df_simple["Code"]
list_to_replace = df_simple["Code"].unique()
list_categories = ['Fitness', 'Health', 'Health', 'Food', 'Food',
                   'Food', 'Others', 'Food', 'Restaurants', 'Restaurants',
                   'Household', 'Transport', 'Mobile', 'Food', 'Food',
                   'Technology', 'Travel', 'Others', 'Household', 'Clothes',
                   'Restaurants', 'Restaurants', 'Food', 'Technology', 'Restaurants', 'Others',
                   'Others', 'Food', 'Others', 'Restaurants', 'Food',
                   'Household', 'Food', 'Health', 'Others',
                   'Others', 'Restaurants', 'Entertainment', 'Others',
                   'Clothes', 'Restaurants', 'Household', 'Education', 'Restaurants',
                   'Transport', 'Mobile', 'Clothes', 'Restaurants',
                   'Health', 'Clothes', 'Entertainment', 'Household', 'Clothes',
                   'Others', 'Clothes', 'Clothes', 'Restaurants', 'Restaurants',
                   'Clothes', 'Restaurants', 'Transport', 'Others', 'Restaurants',
                   'Food', 'Restaurants', 'Food', 'Restaurants', 'Restaurants',
                   'Food', 'Restaurants', 'Restaurants', 'Food',
                   'Food', 'Household', 'Travel', 'Food',
                   'Household', 'Household', 'Travel', 'Restaurants',
                   'Food', 'Restaurants', 'Restaurants', 'Food', 'Restaurants',
                   'Clothes', 'Travel', 'Food', 'Travel', 'Restaurants',
                   'Food', 'Others', 'Others', 'Others', 'Restaurants',
                   'Clothes', 'Others', 'Restaurants', 'Household', 'Health',
                   'Restaurants', 'Health', 'Health', 'Restaurants', 'Restaurants',
                   'Food', 'Travel', 'Others', 'Transport',
                   'Food', 'Restaurants', 'Transport', 'Restaurants', 'Others',
                   'Restaurants', 'Others', 'Transport', 'Restaurants', 'Food',
                   'Restaurants', 'Restaurants', 'Restaurants', 'Restaurants', 'Food',
                   'Rent']
df_simple["Category"] = df_simple["Category"].replace(list_to_replace, list_categories)
# dropping empty values
df_simple = df_simple.dropna()

counts = df_simple["Category"].value_counts()
counts.plot(kind='bar', legend=False, grid=True, figsize=(8, 5))
# plt.show()

lens = df_simple.Code.str.len()
lens.hist(bins=np.arange(0, 20, 0.5))
# plt.show()

# creating subset of data
# for cat in df_simple["Category"]:
#     temp_df = df_simple[df_simple["Category"] == cat][:20]
#     df_simple = pd.concat([df_simple, temp_df])

# loading the models listed at https://github.com/MartinoMensio/spacy-sentence-bert/
nlp = spacy_sentence_bert.load_model('en_stsb_distilbert_base')
df_simple["vector"] = df_simple["Code"].apply(lambda x: nlp(x).vector)

# splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_simple["vector"].tolist(), df_simple["Category"].tolist(),
                                                    test_size=0.33, random_state=42)
# training the model with Support vector classifier
clf = SVC(gamma='auto')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"SVC accuracy: {accuracy_score(y_test, y_pred)}")

# training the model with Random Forest classifier
clf_rf = RandomForestClassifier(max_depth=9, random_state=0)
clf_rf.fit(X_train, y_train)
y_predict = clf_rf.predict(X_test)
print(f"Random forest accuracy: {accuracy_score(y_test, y_predict)}")

# saving the trained model as a pickle file, Random Forest has higher accuracy
pickle.dump(clf_rf, open('model.pkl', 'wb'))
