# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 11:53:34 2018

@author: Shawlock
"""
#SVM
cvec = CountVectorizer(ngram_range=(1, 1),stop_words='english',lowercase = True, max_features = 5000)
cvec.fit(personality['posts'])

#k-folds
kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

 model_svm = Pipeline([('vect', CountVectorizer()),
...                      ('tfidf', TfidfTransformer()),
...                      ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
...                                            alpha=1e-3, random_state=42)),
... ])

scoring1 = {'acc': 'accuracy',
           'f1_micro': 'f1_micro'}

results_svm = cross_validate(model_svm, personality['clean_posts'], personality['type'], cv=kfolds, 
                          scoring=scoring1, n_jobs=-1)

#Accuracy
print("CV Accuracy: {:0.4f} (+/- {:0.4f})".format(np.mean(results_svm['test_acc']),
                                                          np.std(results_svm['test_acc'])))

print("CV F1: {:0.4f} (+/- {:0.4f})".format(np.mean(results_svm['test_f1_micro']),
                                                          np.std(results_svm['test_f1_micro'])))

#Optimization
from sklearn.model_selection import GridSearchCV
>>> parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],
...               'tfidf__use_idf': (True, False),
...               'clf-svm__alpha': (1e-2, 1e-3),
... }
gs_clf = GridSearchCV(model_svm, parameters_svm, n_jobs=-1)
gs_clf = gs_clf.fit(personality['clean_posts'], personality['type'])

gs_clf.best_params_
 
gs_clf.best_score_