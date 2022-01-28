import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pickle

filename = 'multimodels/x_train.p'
file = open(filename, 'rb')
x_train = pickle.load(file)
file.close()
filename = 'multimodels/y_train.p'
file = open(filename, 'rb')
y_train = pickle.load(file)
file.close()
filename = 'multimodels/x_test.p'
file = open(filename, 'rb')
x_test = pickle.load(file)
file.close()
filename = 'multimodels/y_test.p'
file = open(filename, 'rb')
y_test = pickle.load(file)
file.close()

names= ['LogisticRegression_iter_10000','LogisticRegression_iter_100000','LogisticRegression_iter_1000000',
        'LogisticRegression w/ sag', 'LinearDiscriminantAnalysis-svd','LinearDiscriminantAnalysis-lsqr',
        'QuadraticDiscriminantAnalysis','SGDClassifier-hinge','SGDClassifier-log','SGDClassifier-modified_huber',
        'SGDClassifier-squared_hinge','SGDClassifier-perceptron','LinearSVC-ovr','RadiusNeighborsClassifier-2600',
        'DecisionTreeClassifier','GaussianNB', 'LinearSVC-crammer_singer','KNeighborsClassifier-20',]

models = []
import pickle
for i in range(1, len(names) + 1):
    filename = 'models/model' + str(i) + '.p'
    file = open(filename, 'rb')
    model = pickle.load(file)
    file.close()
    models.append(model)
print(len(models))

best_a_test_score = 0
best_a_test_index = 0
best_p_test_score = 0
best_p_test_index = 0
best_r_test_score = 0
best_r_test_index = 0

best_a_train_score = 0
best_a_train_index = 0
best_p_train_score = 0
best_p_train_index = 0
best_r_train_score = 0
best_r_train_index = 0

best_a_all_score = 0
best_a_all_index = 0
best_p_all_score = 0
best_p_all_index = 0
best_r_all_score = 0
best_r_all_index = 0

for i in range(0,len(models)):
    print()
    print('-----------------------------------------------------------------------')
    print()
    print('index: ', i + 1, ' model type: ', names[i])
    model = models[i]
    
    # get predicted y for each:
    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    
    # get accuracies, precisions, recall scores:
    a_test = accuracy_score(y_test, y_pred_test, normalize=True)
    a_test_score = accuracy_score(y_test, y_pred_test, normalize=False)
    a_train = accuracy_score(y_train, y_pred_train, normalize=True)
    a_train_score = accuracy_score(y_train, y_pred_train, normalize=False)
    
    p_test = precision_score(y_test, y_pred_test, average='weighted')
    p_train = precision_score(y_train, y_pred_train, average='weighted')
    
    r_test = recall_score(y_test, y_pred_test, average='weighted')
    r_train = recall_score(y_train, y_pred_train, average='weighted')
    
    # Cross Tabs
    cross_results_test = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred_test})
    result_cross_test = pd.crosstab(cross_results_test.y_pred, cross_results_test.y_test)
    cross_results_train = pd.DataFrame({'y_test': y_train, 'y_pred': y_pred_train})
    result_cross_train = pd.crosstab(cross_results_train.y_pred, cross_results_train.y_test)
    
    # find best a, p, r
    if a_test > best_a_test_score:
        best_a_test_score = a_test
        best_a_test_index = i
    if a_train > best_a_train_score:
        best_a_train_score = a_train
        best_a_train_index = i
        
    if p_test > best_p_test_score:
        best_p_test_score = p_test
        best_p_test_index = i
    if p_train > best_p_train_score:
        best_p_train_score = p_train
        best_p_train_index = i
        
    if r_test > best_r_test_score:
        best_r_test_score = r_test
        best_r_test_index = i
    if r_train > best_r_train_score:
        best_r_train_score = r_train
        best_r_train_index = i
    
    print('accuracies:')
    print('test: ', a_test, ' count: ', a_test_score)
    print('train: ', a_train, ' count: ', a_train_score)
    print('precisions:')
    print('test: ', p_test)
    print('train: ', p_train)
    print('recall:')
    print('test: ', p_test)
    print('train: ', p_train)
    print('Crosstabs')
    print('test:')
    print(result_cross_test)
    print()
    print('train:')
    print(result_cross_train)
    print()