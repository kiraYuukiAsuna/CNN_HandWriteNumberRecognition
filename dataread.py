import os
import numpy as np

class DataRead(object):
    def __init__(self):
        train_data_path = 'data/trainingDigits'
        test_data_path = 'data/testDigits'

        generic_i1 = 0
        #训练集一共1934个
        train_images = np.empty((1934, 32, 32))
        train_labels = np.empty(1934)

        for img_name in os.listdir(train_data_path):
            file = open(train_data_path+'/'+img_name, 'r', encoding='utf-8')
            list1 = []
            list2 = []
            for i in file.readlines():
                for j in i:
                    list1.append(j)
                list1.remove('\n')
                list2.append(list1)
                list1 = []
            file.close()

            train_images[generic_i1] = np.array(list2)

            train_labels[generic_i1] = img_name.split('_')[0]
            generic_i1 = generic_i1+1

        train_images.astype(int)
        train_images = train_images.reshape((1934, 32, 32, 1))

        generic_i2 = 0
        #测试集一共946个
        test_images = np.empty((946, 32, 32))
        test_labels = np.empty(946)

        for img_name in os.listdir(test_data_path):
            file = open(train_data_path+'/'+img_name, 'r', encoding='utf-8')
            list1 = []
            list2 = []
            for i in file.readlines():
                for j in i:
                    list1.append(j)
                list1.remove('\n')
                list2.append(list1)
                list1 = []
            file.close()
            test_images[generic_i2] = np.array(list2)
            #print(img_name)
            # test_images[i]=np.loadtxt(test_data_path+'/'+img_name);
            test_labels[generic_i2] = img_name.split('_')[0]
            generic_i2 = generic_i2+1

        test_images.astype(int)
        test_images = test_images.reshape((946, 32, 32, 1))

        self.train_images, self.train_labels = train_images, train_labels
        self.test_images, self.test_labels = test_images, test_labels


