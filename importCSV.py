import numpy as np


def import_list(filename):
    csv = open('F:\\Datasets\\'+filename)
    values = csv.read()
    values = values.split('\n')
    return values

def import_array(filename):
    values = import_list(filename)

    for i in range(len(values)-1):
        values[i] = values[i].split(';')
        array[values[i][0]] = values[i][1] 

    return array

    

    


label_list = import_list('jester-v1-labels.csv')
label_list = np.asarray(label_list)

testID_list = import_list('jester-v1-test.csv')
testID_list = np.asarray(testID_list)

trainID_list = import_array('jester-v1-train.csv')

validationID_list = import_array('jester-v1-validation.csv')



# int(label_list.shape, testID_list.shape, trainID_list.shape, validationID_list.shape)

# print(testID_list.shape, testID_list)

np.save('DeepLearningModel\\Labels\\testID_list', testID_list)
np.save('DeepLearningModel\\Labels\\label_list', label_list)
