import pickle
import xlsxwriter

runs = 30
c_names = ['function']
for i in range(runs):
    c_names.append('run' + str(i))
c_names.append('fitness min')
c_names.append('fitness max')
c_names.append('fitness avg')

first_part = 'pickles/results_'
to_add = 'TF4_v3_7p' # switch with different models
end_part = '_fits.p'

filename = first_part + to_add + end_part
exampleFile = open(filename, 'rb')
example = pickle.load(exampleFile)
exampleFile.close()

workbook = xlsxwriter.Workbook('excel_files/' + to_add + '.xlsx')
workbook.use_zip64()
worksheet = workbook.add_worksheet(to_add)

row = 0
column = 0

for i in range(len(c_names)):
    worksheet.write(row, column,c_names[i])
    column += 1

row += 1
column = 0

for i in range(len(example)):
    column = 0
    worksheet.write(row, column, i + 1)
    column += 1
    min_f = example[i][0]
    max_f = example[i][0]
    avg_f = 0
    for j in range(len(example[0])):
        temp = example[i][j]
        avg_f += temp
        worksheet.write(row, column, temp)
        column += 1
        if temp < min_f:
            min_f = temp
        if temp > max_f:
            max_f = temp 
    worksheet.write(row, column, min_f)
    column += 1
    worksheet.write(row, column, max_f)
    column += 1
    worksheet.write(row, column, avg_f / 30)
    row += 1
            

workbook.close()

end_part = '_thresholds.p'

filename = first_part + to_add + end_part
exampleFile = open(filename, 'rb')
thresholds = pickle.load(exampleFile)
exampleFile.close()

functions = 28
num_thresholds = 299

ct_names = ['function']
for i in range(runs):
    for j in range(num_thresholds):
        ct_names.append('t-' + str(i) + '-' + str(j))
ct_names

workbookt = xlsxwriter.Workbook('excel_files/ ' + to_add + '_thresh.xlsx')
workbookt.use_zip64()
worksheett = workbookt.add_worksheet(to_add + ' thresh')
rt = 0 
ct = 0

for i in range(len(ct_names)):
    worksheett.write(rt, ct,ct_names[i])
    ct += 1

rt += 1
ct = 0

for i in range(functions):
    ct = 0
    worksheett.write(rt, ct, i + 1)
    ct += 1
    for j in range(runs):
        for k in range(num_thresholds):
            worksheett.write(rt, ct,thresholds[i][j][k])
            ct += 1
    rt += 1


workbookt.close()


ct_names = ['function']
for i in range(runs):
    ct_names.append('run-' + str(i))


workbook = xlsxwriter.Workbook('excel_files/' + to_add + '_avgthresh.xlsx')
workbook.use_zip64()
worksheet = workbook.add_worksheet(to_add)


rt = 0 
ct = 0
for j in range(len(ct_names)):
    worksheet.write(rt, ct,ct_names[j])
    ct += 1

rt += 1
ct = 0

for j in range(functions):
    ct = 0
    worksheet.write(rt, ct, j + 1)
    ct += 1
    for m in range(runs):
        total_t = 0
        for k in range(num_thresholds):
            #worksheet.write(rt, ct,thresholds[i][j][k])
            #ct += 1
            total_t += thresholds[j][m][k]
        avg = total_t / num_thresholds
        worksheet.write(rt, ct,avg)
        ct += 1
    rt += 1



workbook.close()