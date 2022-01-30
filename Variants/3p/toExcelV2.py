import pickle
import xlsxwriter
import pandas as pd 
import os
import sys
import inspect


start_file = 1 # the first number for the part between end_file and ending_file in the file name
end_file = 28 # the last number for the files I have.
init_file = '../../abr data/emna_g_exp' # Where collected data was kept
second_file = '/fun'
mid_file = '_exp'
end_file_name = '.p'
exp_dir_nums = []
functions = []
for i in range(1,11):
    exp_dir_nums.append(str(i))
    functions.append(28)
for i in range(11,20):
    s = str(i) + '-'
    exp_dir_nums.append(s)
functions.append(23)
functions.append(15)
functions.append(18)
functions.append(19)
functions.append(16)
functions.append(23)
functions.append(24)
functions.append(24)
functions.append(24)

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

titles3 = ['exp', 'func', 'number used', 'number excluded', 'extra changed to 5 in variant 2']

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0,parent_dir)

#used for x% difference between high/low
difference = 0.03

workbook = xlsxwriter.Workbook('data_col_v2.xlsx') # First Excel file
workbook.use_zip64()
worksheet = workbook.add_worksheet('main data')
workbook2 = xlsxwriter.Workbook('excel_files/data_info_v2.xlsx') # Info Excel file
worksheet2 = workbook2.add_worksheet('data info')
count  = 0 # Will be incremented to make up the count attribute
row = 0
column = 0
num_gammas = 9
num_multiples = 0
multiples_array = []
total_used = 0
full_set = 0
partial_set = 0
var2to6 = 0
total_used_v2 = 0

# write column headers
for i in range(0, len(titles)):
    worksheet.write(row,column,titles[i])
    column += 1

# Write info headers
rowI = 0
columnI = 0
for i in range(0, len(titles3)):
    worksheet2.write(rowI,columnI,titles3[i])
    columnI += 1
columnI = 0
rowI = 1
    
# Increment Row
column = 0
row = 1

for e in range(1,len(exp_dir_nums) + 1):
    
    for i in range(start_file, functions[e - 1] + 1):
        included = 0
        excluded = 0
        extra_v2_ex = 0
        
        # Get LinkedNode from file
        expString = ''
        if e < 9:
            expString = exp_dir_nums[e-1][0:1]
        else:
            expString = exp_dir_nums[e-1][0:2]
            
        filename = init_file + exp_dir_nums[e-1] + second_file + str(i) + mid_file + expString + end_file_name
        example = open(filename, 'rb')
        exampleContents = pickle.load(example)
        example.close()
        
        for j in range(0, len(exampleContents)):
            max_f = exampleContents[j].max_f
            worksheet.write(row,column,count)
            count += 1
            column += 1
            worksheet.write(row,column,e)
            column += 1
            worksheet.write(row,column,i)
            column += 1
            worksheet.write(row,column,j)
            column += 1
            
            for z in range(0,3):
                worksheet.write(row,column,exampleContents[j].data[z][0])
                column += 1
                
            # Threshold done separately
            for z in range(0,len(exampleContents[j].data[3])):
                worksheet.write(row,column,exampleContents[j].data[3][z])
                column += 1

            # starting at 4 to exclude threshold
            for k in range(4,len(exampleContents[j].data)):
                for z in range(0,len(exampleContents[j].data[4])):
                    worksheet.write(row,column,exampleContents[j].data[k][z] / max_f)
                    column += 1
                    
            best_g = 0
            best_gs = '0'
            best_f = 1000000000
            worst_f = -1000000000
            range_min = 1000000000
            range_max = 1000000000
            range_avg = 1000000000
            
            for w in range(0, len(exampleContents[j].results)):
                f = exampleContents[j].results[w].best_f
                if f > worst_f:
                    worst_f = f
                
                if abs(f - range_avg) < 0.001:
                    best_gs = best_gs + ' ' + str(float(exampleContents[j].results[w].gamma))
                    if f < best_f:
                        best_f = f
                        best_g = float(exampleContents[j].results[w].gamma)
                    if f < range_min:
                        range_min = f
                        range_avg = (range_max + range_min) / 2
                    elif f > range_max:
                        range_max = f
                        range_avg = (range_max + range_min) / 2
                elif f < best_f:
                    best_f = f
                    range_min = f
                    range_max = f
                    range_avg = f
                    
                    best_g = int(exampleContents[j].results[w].gamma * 2)
                    best_gs = str(best_g)
                    
                worksheet.write(row,column,f)
                column += 1
            worksheet.write(row,column,best_f)
            column += 1
            worksheet.write(row,column,best_g)
            column += 1
            worksheet.write(row,column,best_gs)
            column = 0

            # This if/else collects information on if the example has equal fitness for all gammas, if so that training set
            # gets overwritten
            if len(best_gs) == 17:
                num_multiples += 1
                multiples_array.append(best_gs)
                
                # for info
                excluded += 1
                full_set += 1
            elif len(best_gs) > 1:
                num_multiples += 1
                multiples_array.append(best_gs)
                
                # for info
                excluded += 1
                partial_set += 1
            else:
                if (abs((best_f - worst_f) / worst_f) > difference):
                    worksheet.write(row,column,best_g)
                else:
                    extra_v2_ex += 1
                    var2to6 += 1
                    worksheet.write(row,column,5)
                    
                total_used += 1
                column += 1
                worksheet.write(row,column,best_gs)
                column = 0
                row += 1
                total_used_v2 += 1
                
                # for info
                included += 1
       
        worksheet2.write(rowI,columnI,e)
        columnI += 1
        worksheet2.write(rowI,columnI,i)
        columnI += 1
        worksheet2.write(rowI,columnI,included)
        columnI += 1
        worksheet2.write(rowI,columnI,excluded)
        columnI += 1
        worksheet2.write(rowI,columnI,extra_v2_ex)
        columnI += 1
        
        rowI += 1
        columnI = 0
        
workbook.close()
workbook2.close()

switchFile = pd.read_excel(r'data_col_v2.xlsx')
switchFile.to_csv(r'excel_files/data_col_v2.csv', index=None, header=True)

os.remove('data_col_v2.xlsx')