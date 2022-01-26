'''
    An example of training all SKLearn models using the
    earlier data, and saving the trained models to file
'''

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import pickle

titles = ['evaluations consumed', 'evaluations available', 'threshold-0']
for i in range(1,100):
    x = 'threshold-' + str(i)
    titles.append(x)
for i in range(0,100):
    x = 'stdev-' + str(i)
    titles.append(x)
for i in range(0,100):
    x = 'avfit-' + str(i)
    titles.append(x)
for i in range(0,100):
    x = '0%-' + str(i)
    titles.append(x)
for i in range(0,100):
    x = '10%-' + str(i)
    titles.append(x)
for i in range(0,100):
    x = '20%-' + str(i)
    titles.append(x)
for i in range(0,100):
    x = '30%-' + str(i)
    titles.append(x)
for i in range(0,100):
    x = '40%-' + str(i)
    titles.append(x)
for i in range(0,100):
    x = '50%-' + str(i)
    titles.append(x)
for i in range(0,100):
    x = '60%-' + str(i)
    titles.append(x)
for i in range(0,100):
    x = '70%-' + str(i)
    titles.append(x)
for i in range(0,100):
    x = '80%-' + str(i)
    titles.append(x)
for i in range(0,100):
    x = '90%-' + str(i)
    titles.append(x)
for i in range(0,100):
    x = '100%-' + str(i)
    titles.append(x)
titles.append('best_gamma')

count = 1

filename = 'v1_models/x_train_v1.p'
file = open(filename, 'rb')
x_train = pickle.load(file)
file.close()
filename = 'v1_models/y_train_v1.p'
file = open(filename, 'rb')
y_train = pickle.load(file)
file.close()
filename = 'v1_models/x_test_v1.p'
file = open(filename, 'rb')
x_test = pickle.load(file)
file.close()
filename = 'v1_models/y_test_v1.p'
file = open(filename, 'rb')
y_test = pickle.load(file)
file.close()

def build_model(model_function):
    model = model_function(x_train, y_train)
    import pickle
    
    model_file = 'v1_models/model' + str(count) + '.p'
    file = open(model_file, 'wb')
    pickle.dump(model, file)
    file.close()

def log_reg1(x_train, y_train, solver='sag', multi_class='auto', max_iter=10000):
    model = LogisticRegression(solver=solver, multi_class=multi_class, max_iter=max_iter)
    model.fit(x_train, y_train)
    return model

build_model(log_reg1)

def log_reg2(x_train, y_train, solver='sag', multi_class='auto', max_iter=100000):
    model = LogisticRegression(solver=solver, multi_class=multi_class, max_iter=max_iter)
    model.fit(x_train, y_train)
    return model

count += 1
build_model(log_reg2)

def log_reg3(x_train, y_train, solver='sag', multi_class='auto', max_iter=1000000):
    model = LogisticRegression(solver=solver, multi_class=multi_class, max_iter=max_iter)
    model.fit(x_train, y_train)
    return model

count += 1
build_model(log_reg3)

def log_fn(x_train, y_train):
    model = LogisticRegression(solver='sag')
    model.fit(x_train, y_train)
    return model

count += 1
build_model(log_fn)

def lda_svd(x_train, y_train, solver='svd'):
    model = LinearDiscriminantAnalysis(solver = solver)
    model.fit(x_train, y_train)
    return model

count += 1
build_model(lda_svd)

def lda_lsqr(x_train, y_train, solver='lsqr'):
    model = LinearDiscriminantAnalysis(solver = solver)
    model.fit(x_train, y_train)
    return model

count += 1
build_model(lda_lsqr)

def qda(x_train, y_train):
    model = QuadraticDiscriminantAnalysis()
    model.fit(x_train, y_train)
    return model

count += 1
build_model(qda)

def sgd_hinge(x_train, y_train, max_iter = 10000, tol=1e-3, loss='hinge'):
    model = SGDClassifier(max_iter=max_iter, tol=tol, loss=loss)
    model.fit(x_train, y_train)
    return model

count += 1
build_model(sgd_hinge)

def sgd_log(x_train, y_train, max_iter = 10000, tol=1e-3, loss='log'):
    model = SGDClassifier(max_iter=max_iter, tol=tol, loss=loss)
    model.fit(x_train, y_train)
    return model

count += 1
build_model(sgd_log)

def sgd_mh(x_train, y_train, max_iter = 10000, tol=1e-3, loss='modified_huber'):
    model = SGDClassifier(max_iter=max_iter, tol=tol, loss=loss)
    model.fit(x_train, y_train)
    return model

count += 1
build_model(sgd_mh)

def sgd_sh(x_train, y_train, max_iter = 10000, tol=1e-3, loss='squared_hinge'):
    model = SGDClassifier(max_iter=max_iter, tol=tol, loss=loss)
    model.fit(x_train, y_train)
    return model

count += 1
build_model(sgd_sh)

def sgd_p(x_train, y_train, max_iter = 10000, tol=1e-3, loss='perceptron'):
    model = SGDClassifier(max_iter=max_iter, tol=tol, loss=loss)
    model.fit(x_train, y_train)
    return model

count += 1
build_model(sgd_p)

def lsvc_ovr(x_train, y_train, C=1.0, max_iter=10000, tol=1e-3, multi_class='ovr'):
    model = LinearSVC(C=C, max_iter=max_iter, tol=tol, dual=False, multi_class=multi_class)
    model.fit(x_train, y_train)
    return model

count += 1
build_model(lsvc_ovr)

def near_n(x_train, y_train, radius=2600.0):
    model = RadiusNeighborsClassifier(radius=radius, outlier_label='most_frequent')
    model.fit(x_train, y_train)
    return model

count += 1
build_model(near_n)

def d_tree(x_train, y_train, max_depth=None, max_features=None):
    model = DecisionTreeClassifier(max_depth=max_depth, max_features=max_features)
    model.fit(x_train, y_train)
    return model

count += 1
build_model(d_tree)

def naive_b(x_train, y_train, priors=None):
    model = GaussianNB(priors=priors)
    model.fit(x_train, y_train)
    return model

count += 1
build_model(naive_b)

def lsvc_cs(x_train, y_train, C=1.0, max_iter = 10000, tol=1e-3, multi_class='crammer_singer'):
    model = LinearSVC(C=C, max_iter=max_iter, tol=tol, dual=False, multi_class=multi_class)
    model.fit(x_train, y_train)
    return model

count += 1
build_model(lsvc_cs)

from sklearn.neighbors import KNeighborsClassifier

def k_near_n(x_train, y_train, n_neighbors=20, weights='distance'):
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    model.fit(x_train, y_train)
    return model

count += 1
build_model(k_near_n)
