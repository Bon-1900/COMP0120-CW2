
import warnings
warnings.filterwarnings('ignore')
import re
import itertools
import string
import time
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import svm
import seaborn as sns

data = pd.read_csv('./data/spam_msgs_dataset/spam_msgs_dataset.csv')
#stopword_list = [k.strip() for k in open("E:/MaLearning/souhu/stopwords.txt", encoding='utf8').readlines() if k.strip() != '']
stopword_list = stopwords.words('english')

# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)`
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()

## Preprocessing
data["Category"] = data["Category"].map({'ham': 0,'spam':1})

description_list = []
for article in data ["Message"]:
        article = re.sub ("[^a-zA-Z]", " ", article)
        article = article.lower ()  # low case letter
        article = word_tokenize (article)
        lemma = WordNetLemmatizer ()
        article = [lemma.lemmatize (word) for word in article]
        article = " ".join (article)
        description_list.append (article)  # we hide all word one section


def text_replace ( text ):
        '''some text cleaning method'''
        text = text.lower ()
        text = re.sub ('\[.*?\]', '', text)
        text = re.sub ('https?://\S+|www\.\S+', '', text)
        text = re.sub ('<.*?>+', '', text)
        text = re.sub ('[%s]' % re.escape (string.punctuation), '', text)
        text = re.sub ('\n', '', text)
        text = re.sub ('\w*\d\w*', '', text)
        return text

count_vectorizer = CountVectorizer(max_features = 100, stop_words = "english")
sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()
tokens = count_vectorizer.get_feature_names()

sparce_matrix = pd.DataFrame(sparce_matrix, columns=tokens)
# sparce_matrix.head()

########### TF-IDF
vectorizer = TfidfVectorizer(max_features = 100)
tfidfmatrix = vectorizer.fit_transform(description_list)
cname = vectorizer.get_feature_names()
tfidfmatrix = pd.DataFrame(tfidfmatrix.toarray(),columns=cname)
# tfidfmatrix.head()
# tfidfmatrix.columns

count_vectorizer = CountVectorizer(max_features = 100, stop_words = "english",ngram_range=(2, 2),)
sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()
tokens = count_vectorizer.get_feature_names()
gram2 = pd.DataFrame(sparce_matrix, columns=tokens)
# gram2.head()

########### SVM
y = data.iloc[:,0].values
x = sparce_matrix
tfidfx = tfidfmatrix

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 2019)
tf_x_train, tf_x_test, tf_y_train, tf_y_test = train_test_split(tfidfmatrix ,y,
                                                                test_size = 0.3,
                                                                random_state = 2019)

gm_x_train, gm_x_test, gm_y_train, gm_y_test = train_test_split(gram2 ,y,
                                                                test_size = 0.3,
                                                                random_state = 2019)

start_time = time.time()
# svmmodel = svm.SVC(kernel='linear', C = 1)
# svmmodel.fit(x_train, y_train)
# print('CountVectorizer Accuracy Score',svmmodel.score(x_test,y_test))
# svmmodel.fit(tf_x_train, tf_y_train)
# print('TF-IDF Vectorizer Accuracy Score',svm model.score(tf_x_test,tf_y_test))
# svmmodel.fit(gm_x_train, gm_y_train)
# print('bi-gram Vectorizer Accuracy Score',svmmodel.score(gm_x_test,gm_y_test))

# svmmodel = svm.SVC(kernel='linear', C = 1)
svmmodel = svm.SVC(kernel='rbf', gamma = 1)
svmmodel.fit(tf_x_train, tf_y_train)
# print('TF-IDF Vectorizer Accuracy Score',svmmodel.score(tf_x_test,tf_y_test))
end_time = time.time()
print(f'Time elapsed: {end_time - start_time}')

tf_y_predict = svmmodel.predict(tf_x_test)
print('TF-IDF Vectorizer Accuracy Score',accuracy_score(tf_y_test, tf_y_predict))
cf_matrix = pd.DataFrame(confusion_matrix(tf_y_test,tf_y_predict))

sns.heatmap((cf_matrix / np.sum(cf_matrix)*100), annot = True, fmt=".2f", cmap="Blues")
plt.scatter(x_train,y_train)
plt.title('Confusion Matrix')

# plot the line, the points, and the nearest vectors to the plane

# Z = svmmodel.decision_function (np.c_ [XX.ravel (), YY.ravel ()])
# plt.figure (2, figsize = (4, 3))
# plt.svmmodel ()

# plt.scatter (
#     svmmodel.support_vectors_ [:, 0],
#     svmmodel.support_vectors_ [:, 1],
#     s = 80,
#     facecolors = "none",
#     zorder = 10,
#     edgecolors = "k",
# )
# Z = svmmodel.decision_function (np.c_[y_test, x_test])
# plt.scatter (X [:, 0], X [:, 1], c = Y, zorder = 10, cmap = plt.cm.Paired, edgecolors = "k")

# plt.axis ("tight")
# x_min = -3
# x_max = 3
# y_min = -3
# y_max = 3
#
# # Put the result into a color plot
# Z = Z.reshape (XX.shape)
# plt.figure (2, figsize = (4, 3))
# plt.pcolormesh (XX, YY, Z > 0, cmap = plt.cm.Paired)
# plt.contour (
#     XX,
#     YY,
#     Z,
#     colors = ["k", "k", "k"],
#     linestyles = ["--", "-", "--"],
#     levels = [-0.5, 0, 0.5],
# )
#
# plt.xlim (x_min, x_max)
# plt.ylim (y_min, y_max)
#
# plt.xticks (())
# plt.yticks (())



plt.show()
# plot_confusion_matrix(conf_matrix,[0,1])




