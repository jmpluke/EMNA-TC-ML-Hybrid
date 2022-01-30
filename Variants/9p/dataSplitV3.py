import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_csv('excel_files/data_col_cleaned_v3.csv')

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


X = data[titles[:-1]]
Y = data['best_gamma']

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2)


model_file = 'v3_models/x_train_v3.p'
file = open(model_file, 'wb')
pickle.dump(x_train, file)
file.close()
model_file = 'v3_models/y_train_v3.p'
file = open(model_file, 'wb')
pickle.dump(y_train, file)
file.close()
model_file = 'v3_models/x_test_v3.p'
file = open(model_file, 'wb')
pickle.dump(x_test, file)
file.close()
model_file = 'v3_models/y_test_v3.p'
file = open(model_file, 'wb')
pickle.dump(y_test, file)
file.close()
model_file = 'v3_models/x_all_v3.p'
file = open(model_file, 'wb')
pickle.dump(X, file)
file.close()
model_file = 'v3_models/y_all_v3.p'
file = open(model_file, 'wb')
pickle.dump(Y, file)
file.close()

