import pandas as pd

data = pd.read_csv('excel_files/data_col_v1.csv')

titles = ['count', 'exp', 'function', 'j', 'evaluations consumed', 'evaluations available', 'gamma', 'threshold-0']
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
    
titles2 = [ 'best_f(0)', 'best_f(0.5)', 'best_f(1)', 
          'best_f(1.5)', 'best_f(2)', 'best_f(2.5)', 'best_f(3)', 'best_f(3.5)', 'best_f(4)', 'best_f',
           'best_gamma', 'best_gammas']
for i in range(0, len(titles2)):
    titles.append(titles2[i])

# Remove unnecessary attributes

data_post = data.drop(['exp', 'function','j','count'], axis=1)

data_post = data_post.drop(['gamma','best_f(0)', 'best_f(0.5)', 'best_f(1)', 
          'best_f(1.5)', 'best_f(2)', 'best_f(2.5)', 'best_f(3)', 'best_f(3.5)', 'best_f(4)', 'best_f', 'best_gammas'], axis=1)

data_post.to_csv('excel_files/data_col_cleaned_v1.csv', index=False)