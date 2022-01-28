import pickle
import xlsxwriter

runs = 30


first_part = 'pickles/results_'
to_add = '' # switch with different models
end_part = '_thresholds.p'
functions = 28
num_thresholds = 299

ct_names = ['function']
for i in range(runs):
    ct_names.append('run-' + str(i))

to_adds = ['combo', 'QDA_org', 'LDA_SVD_org', 'RNN_org', 'DT_org', 'EMNA_TC']
q = 'QDA_'
l = 'LDA_SVD_'
r = 'RNN_'
d = 'DT_'
mains = [q,l,r,d]
posts = ['v2_1p','v2_5p','v2_9p']
post_dt = ['v3_1p','v3_5p','v3_9p']
for i in range(len(mains)):
    for j in range(len(posts)):
        to_adds.append(mains[i] + posts[j])
for i in range(len(post_dt)):
    to_adds.append('DT_' + post_dt[i])

for i in range(len(to_adds)):
    to_add = to_adds[i]
    workbook = xlsxwriter.Workbook('excel_files/' + to_add + '_avgthresh.xlsx')
    workbook.use_zip64()
    worksheet = workbook.add_worksheet(to_add)
    
    rt = 0 
    ct = 0

    filename = first_part + to_add + end_part
    exampleFile = open(filename, 'rb')
    thresholds = pickle.load(exampleFile)
    exampleFile.close()

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