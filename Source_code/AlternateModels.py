# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 14:43:40 2018

@author: Shawlock
"""

#Let's try if we can find a more accuracy model, although we haven't got a lot of data

#Text Cleaning
def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    return text

personality['clean_posts'] = personality['posts'].apply(cleanText)


# NAIVE BAYES 
np.random.seed(1)

tfidf2 = CountVectorizer(ngram_range=(1, 1), 
                         stop_words='english',
                         lowercase = True, 
                         max_features = 5000)

model_nb = Pipeline([('tfidf1', tfidf2), ('nb', MultinomialNB())])

results_nb = cross_validate(model_nb, personality['clean_posts'], personality['type'], cv=kfolds, 
                          scoring=scoring, n_jobs=-1)


#Accuracy
print("CV Accuracy: {:0.4f} (+/- {:0.4f})".format(np.mean(results_nb['test_acc']),
                                                          np.std(results_nb['test_acc'])))

print("CV F1: {:0.4f} (+/- {:0.4f})".format(np.mean(results_nb['test_f1_micro']),
                                                          np.std(results_nb['test_f1_micro'])))

print("CV Logloss: {:0.4f} (+/- {:0.4f})".format(np.mean(-1*results_nb['test_neg_log_loss']),
                                                          np.std(-1*results_nb['test_neg_log_loss'])))





#LOGISTIC REGRESSION
np.random.seed(1)

tfidf2 = CountVectorizer(ngram_range=(1, 1), stop_words='english',
                                                 lowercase = True, max_features = 5000)

model_lr = Pipeline([('tfidf1', tfidf2), ('lr', LogisticRegression(class_weight="balanced", C=0.005))])

results_lr = cross_validate(model_lr, personality['clean_posts'], personality['type'], cv=kfolds, 
                          scoring=scoring, n_jobs=-1)


#Accuracy
print("CV Accuracy: {:0.4f} (+/- {:0.4f})".format(np.mean(results_lr['test_acc']),
                                                          np.std(results_lr['test_acc'])))

print("CV F1: {:0.4f} (+/- {:0.4f})".format(np.mean(results_lr['test_f1_micro']),
                                                          np.std(results_lr['test_f1_micro'])))

print("CV Logloss: {:0.4f} (+/- {:0.4f})".format(np.mean(-1*results_lr['test_neg_log_loss']),
                                                          np.std(-1*results_lr['test_neg_log_loss'])))

#PARAMETER TUNING for Logistic Regression(via GridCV)
from sklearn.model_selection import GridSearchCV

parameters_lr = {'tfidf1__ngram_range': [(1, 1), (1, 2)],
...               'lr__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
... }
grid = GridSearchCV(model_lr, parameters_lr, cv=5, scoring = 'accuracy', )
grid.fit(personality['posts'], personality['types'])

grid.best_score_

grid.best_params_