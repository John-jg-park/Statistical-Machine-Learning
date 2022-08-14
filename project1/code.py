
# coding: utf-8

# In[45]:


import numpy
import scipy.io
import math
import geneNewData
from math import sqrt
from math import pi
from math import exp

def main():
    myID='0471'
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID+'.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID+'.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset'+'.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset'+'.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')
    print([len(train0),len(train1),len(test0),len(test1)])
    print('Your trainset and testset are generated successfully!')

    #-------------------Part 1----------------------------
    
    #print(len(train0))
    #print(type(train0))
    # initialize numpy arrays to hold average and std. deviation 
    train0_features = numpy.zeros([len(train0), 2])
    train1_features = numpy.zeros([len(train1), 2])

    # calculate average and std in Train 0 
    i=0
    for sample in train0:
        train0_features[i, 0] = numpy.mean(sample)
        train0_features[i, 1] = numpy.std(sample)
        i += 1
    
    # calculate average and std in Train 1 
    i=0
    for sample in train1:
        train1_features[i, 0] = numpy.mean(sample)
        train1_features[i, 1] = numpy.std(sample)
        i += 1   

    #print(train0_features, train1_features)
    #for sample in train0_features:
    #    print(sample)
    
    #-------------------Part 2----------------------------   
    train0_means = numpy.mean(train0_features, axis = 0)
    #rint(train0_means)
    train0_vars = numpy.var(train0_features, axis = 0)
    #print(train0_vars)
    print('Mean_of_feature1_for_digit0=      ', train0_means[0])
    print('Variance_of_feature1_for_digit0 = ', train0_vars[0])
    print('Mean_of_feature2_for_digit0 =     ', train0_means[1])
    print('Variance_of_feature2_for_digit0 = ', train0_vars[1])
    train1_means = numpy.mean(train1_features, axis = 0)
    #print(train1_means)
    train1_vars = numpy.var(train1_features, axis = 0)
    #print(train1_vars)    
    print('Mean_of_feature1_for_digit1 =     ', train1_means[0])
    print('Variance_of_feature1_for_digit1 = ', train1_vars[0])
    print('Mean_of_feature2_for_digit1 =     ', train1_means[1])
    print('Variance_of_feature2_for_digit1 = ', train1_vars[1])
    
    #-------------------Part 3----------------------------     
        
    # Calculate the Gaussian probability distribution function for x
    def cal_gauss_prob(x, mean, var):
        return (1 / (sqrt(2 * pi * var)) * exp(-((x-mean)**2 / (2 * var))))
    #print(cal_gauss_prob(1,1,4))
    #print(cal_gauss_prob(2,1,4))
    
    # Test data arrays
    # initialize numpy arrays to hold average and std. deviation 
    test0_features = numpy.zeros([len(test0), 2])
    test1_features = numpy.zeros([len(test1), 2])

    # calculate average and std in Train 0 
    i=0
    for sample in test0:
        test0_features[i, 0] = numpy.mean(sample)
        test0_features[i, 1] = numpy.std(sample)
        i += 1
    
    # calculate average and std in Train 1 
    i=0
    for sample in test1:
        test1_features[i, 0] = numpy.mean(sample)
        test1_features[i, 1] = numpy.std(sample)
        i += 1     
    
    #print(test0_features)
    #print(test1_features)

    # predicting label for test 0 data
    i=0 #predicting label digit 0 in test data set
    t=0 #total number of test data set
    for test in test0_features:
        #print(test[0], test[1])
        prob_0 = 0.5 * cal_gauss_prob(test[0], train0_means[0], train0_vars[0]) * cal_gauss_prob(test[1], train0_means[1], train0_vars[1])
        prob_1 = 0.5 * cal_gauss_prob(test[0], train1_means[0], train1_vars[0]) * cal_gauss_prob(test[1], train1_means[1], train1_vars[1])
        #print(prob_0, prob_1)
        if prob_0 > prob_1:
            i += 1
        t += 1 
    #print(i, t)
    print('Accuracy_for_digit0testset = ', i/t)
    
    # predicting label for test 1 data
    i=0 #predicting label digit 1 in test data set
    t=0 #total number of test data set
    for test in test1_features:
        #print(test[0], test[1])
        prob_0 = 0.5 * cal_gauss_prob(test[0], train0_means[0], train0_vars[0]) * cal_gauss_prob(test[1], train0_means[1], train0_vars[1])
        prob_1 = 0.5 * cal_gauss_prob(test[0], train1_means[0], train1_vars[0]) * cal_gauss_prob(test[1], train1_means[1], train1_vars[1])
        #print(prob_0, prob_1)
        if prob_0 < prob_1:
            i += 1
        t += 1 
    #print(i, t)
    print('Accuracy_for_digit1testset = ', i/t)
    
        
    pass


if __name__ == '__main__':
    main()

   


# In[ ]:




