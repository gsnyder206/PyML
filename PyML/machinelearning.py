#!/usr/bin/env python

"""
My own personal Machine Learning codes.

Includes PCA, Diffusion Mapping, Random Forest
"""

__author__ = "Michael Peth, Peter Freeman"
__copyright__ = "Copyright 2015"
__credits__ = ["Michael Peth", "Peter Freeman"]
__license__ = "GPL"
__version__ = "0.3.1"
__maintainer__ = "Michael Peth"
__email__ = "mikepeth@pha.jhu.edu"

from numpy import *
import os
#import pyfits
#from pygoods import *
import scipy
from scipy.sparse.linalg import eigsh
from sklearn.ensemble import RandomForestClassifier
import pickle
import PyML
import pandas as pd


def whiten(data, A_basis=False):
    '''
    Whitens the input data set by subtracting the mean and normalizing by the standard deviation for each feature

    Parameters
    ----------
    data: matrix
    Input data (Nxk): N objects by k features


    Returns
    -------
    whiten_data: matrix
    Input data that has been average subtracted and stddev scaled (Nxk)
    '''

    #Take data (Nxk) N objects by k features, subtract mean, and scale by variance
    
    
    
    
    if A_basis == False:
        mu = mean(data,axis=0)
        wvar = std(data,axis=0)
    else:
        whiten_path=PyML.__path__[0]+os.path.sep+"data"+os.path.sep+"candels_whiten_j_avg_std.txt"
        #with open(whiten_path, 'rb') as handle:
        #    a_basis = pickle.loads(handle.read())
        a_basis = load(whiten_path,encoding='bytes')
        mu = a_basis[b'mean']
        wvar = a_basis[b'std']

    whiten_data = zeros(shape(data))
    for p in range(len(mu)):
        whiten_data[:,p]  = (data[:,p] - mu[p])/wvar[p]
    return whiten_data

def trainingSet(df,training_fraction=0.67):
    '''
    Create training set and test set for random forest

    Parameters
    ----------
    df: DataFrame
    Input catalog


    Returns
    -------
    df_67 = Randomly selected Training set 
    df_33 = Randomly selected Test set (does not contain training set)
    '''

    import random

    rows = random.sample(list(df.index), int(len(df)*training_fraction))

    df_67 = df.ix[rows] #training set

    df_33 = df.drop(rows) #test set

    return df_67, df_33

def morphMatrix(data,band='J'):
    '''
    Create an Nxk matrix to be used for PCA
        ['C','M20','G','A','M','I','D']

    Parameters
    ----------
    data: FITS rec or dictionary
    Input catalog


    Returns
    -------
    new_matrix: matrix
        Nxk matrix to be used for PCA

    Re-write to eliminate function
    '''

    new_matrix = zeros((len(data['GINI_%s' % band]),7))

    new_matrix[:,0] = data['C_%s' % band]
    new_matrix[:,1] = data['M20_%s' % band]
    new_matrix[:,2] = data['GINI_%s' % band]
    new_matrix[:,3] = data['ASYM_%s' % band]
    new_matrix[:,4] = data['MPRIME_%s' % band]
    new_matrix[:,5] = data['I_%s' % band]
    new_matrix[:,6] = data['D_%s' % band]
    return new_matrix

def dataMatrix(data,parameter_list):
    '''
    Create an Nxk matrix to be used for PCA/Random Forest

    Parameters
    ----------
    data:           FITS rec or dictionary
                    Input catalog
    parameter_list: list of parameters to include

    (To use PCs from Peth et al. 2015 use ['C','M20','GINI','ASYM','MPRIME','I','D'] as input parameter list)


    Returns
    -------
    new_matrix: matrix
        Nxk matrix to be used for PCA

    '''

    new_matrix = zeros((len(data[parameter_list[0]]),len(parameter_list)))

    for pl in range(len(parameter_list)):
        new_matrix[:,pl] = data[parameter_list[pl]]

    return new_matrix



class PCA:
    def __init__(self, data):
        '''
        Compute a Principal Component analysis p for a data set

        Parameters
        ----------
        whiten_data: matrix
        Input data (Nxk): N objects by k features


        Returns
        -------
        Structure with the following keys:

        X: matrix
        Principal Component Coordinates

        values: array
        Eigenvalue solutions to SVD

        vectors: matrix
        Eigenvector solutions to SVD
        '''
        whiten_data = whiten(data)

        #Calculate eigenvalues/eigenvectors from SVD
        u, s, v = linalg.svd(whiten_data)

        #Force eigenvalues between 0 and 1
        eigenvalues = s**2/sum(s**2)

        #Change data to PC basis
        pc = zeros(shape(whiten_data))
        for i in range(len(whiten_data[0])):
            for j in range(len(whiten_data[0])):
                pc[:,i] = pc[:,i] + v[i][j]*whiten_data[:,j]

        self.X = pc
        self.values = eigenvalues
        self.vectors = v

        return

class pcV:
    def __init__(self,data):
        '''
        Compute a Principal Component analysis p for a data set

        Parameters
        ----------
        data: matrix
        Input data (Nxk): N objects by k features

        Returns
        -------
        Structure with the following keys:

        X: matrix
        Principal Component Coordinates

        Default Import
        -------
        A_pcv: Matrix
        Data (NxK) with Eigenvector solutions used to project data (from CANDELS Morphologies)

        '''
        #npmorph_path="PC_f125w_candels.txt" 
        #npmorph_path=PyML.__path__[0]+os.path.sep+"data"+os.path.sep+"npmorph_f125w_candels.txt"
        npmorph_path=PyML.__path__[0]+os.path.sep+"data"+os.path.sep+"PC_f125w_candels.txt"
        #with open(npmorph_path, 'rb') as handle:
        #    pc1 = pickle.loads(handle.read())
        pc1 = load(npmorph_path,encoding='bytes')

        whiten_data = whiten(data,A_basis=True)
        pc = zeros(shape(whiten_data))
        
        for i in range(len(whiten_data[0])):
             for j in range(len(whiten_data[0])):
                pc[:,i] = pc[:,i] + pc1[b'vectors'][i][j]*whiten_data[:,j]

        self.X = pc
        self.vectors = pc1[b'vectors']
        self.values = pc1[b'values']
        return

class diffusionMap:
    def __init__(self, data, epsilon=0.2,delta=1e-10,n_eig=100):
        '''
        Compute a diffusion Map for a data set, based heavily on
        Lee & Freeman 2012

        Parameters
        ----------
        data: matrix
        Input data (Nxk): N objects by k features, not whitened

        epsilon: float
        value determined to optimize function, default is 0.2

        delta: float
        delta: minimum value used in e^(-d^2/eps) matrix, creates sparse matrix

        n_eig: int
        Number of eigenvectors to keep, default is 100

        Returns
        -------
        Structure with the following keys:

        X: matrix
        Diffusion Map Coordinates = weighted eigenvectors * eigenvalues

        eigenvals: array
        Eigenvalue solutions to Diffusion problem

        psi: matrix
        Weighted eigenvectors

        weights: array
        First eigenvector
        '''
        distance = zeros((len(data),len(data))) #NxN distance function

        #Step 1. Build matrix (NxN) of distances using e^(-d_ij^2/epsilon), d_ij = Euclidean distance
        for i in range(len(data)): #N objects
            for j in range(len(data)): #N-i objects
                distance_sq = scipy.spatial.distance.euclidean(data[i],data[j])*scipy.spatial.distance.euclidean(data[i],data[j])
                distance[i][j] = math.exp(-1.0*distance_sq/epsilon)
                if distance[i][j] < delta:
                    distance[i][j] = 0.0 #Removes values that are very small

        k = sqrt(sum(distance,axis=1)).reshape((len(distance),1)) #sqrt of the sum of the rows
        A = distance/(inner(k,k))

        #Step 2. Eigendecompose distance matrix
        N = shape(data)[0]
        l, v = linalg.eig(A)

        #Sort eigenvectors based on eigenvalues
        l_sort = sorted(l,reverse=True)
        sort_indx = argsort(l)[::-1]

        #Place sorted eigenvectors into new matrix
        eigenvector_sort = zeros(shape(v))
        for indx in range(len(sort_indx)):
            eigenvector_sort[indx] = v[sort_indx][indx]

        #Create matrix containing eigenvalues
        eigenvalues = array((l_sort[1:])).reshape(len(l_sort[1:]),1)
        lambda_x = inner(ones((N,1)),eigenvalues)

        #Psi = Eigenvectors/eigenvector[1]
        weight = inner(eigenvector_sort[:,0].reshape(N,1),ones((N,1)))
        psi = eigenvector_sort/weight

        #Step 3. Project original data onto newly defined basis
        X = psi[:,1:n_eig]*lambda_x[:,0:n_eig-1]

        self.X = X
        self.eigenvals = l_sort
        self.psi = psi
        self.weight = weight

        return

def confusionMatrix(df,probs,threshold=0.5,traininglabel='mergerFlag'):
    """
    Calculate Summary Statistics (completeness,specificity,risk,error, PPV,NPV)

    Parameters
    ----------
    df: DataFrame pandas
    Basis catalog for which to Random Forest is trained upon

    rfmc: function
    Predicted labels for galaxies in df
    
    probs:  predicted probabilities


    Output
    ----------
    summaryStats: dictionary
    Summary Statistics (completeness,specificity,risk,error, PPV,NPV)

    """

    tp = len(where((df[traininglabel].values == 1) & (probs[:,1] >= threshold))[0])
    fp = len(where((df[traininglabel].values == 0) & (probs[:,1] >= threshold))[0])
    fn = len(where((df[traininglabel].values == 1) & (probs[:,1] < threshold))[0])
    tn = len(where((df[traininglabel].values == 0) & (probs[:,1] < threshold))[0])

    Ntp= len(where(df[traininglabel].values == 1)[0])
    Ntn= len(where(df[traininglabel].values == 0)[0])

    
    completeness = 1.*tp/(tp+fn)
    specificity = 1.*tn/(tn+fp)
    risk = (1 - completeness) + (1 - specificity)
    error = 1.*(fn+fp)/(tn+fp+tp+fn)
    if tp+fp == 0:
        ppv = 0
    else:
        ppv = 1.*tp/(tp+fp)

    if tn+fn ==0:
        npv=1.0
    else:
        npv = 1.*tn/(tn+fn)

    tpr=(1.0*tp)/float(Ntp)
    fpr=(1.0*fp)/float(Ntn)
    
    summaryStats = {'completeness': completeness, 'specificity': specificity,'risk':risk,'totalError':error,'ppv':ppv,'npv':npv,'tpr':tpr,'fpr':fpr,'thresh':threshold}
    return summaryStats

class randomForest:
    def __init__(self,df,cols=None,trainDF=None,testDF=None, traininglabel='mergerFlag',trainfrac=0.67,**kwargs):
        """
        Parameters
        ----------
        df: DataFrame pandas
        Basis catalog for which to create Random Forest

        n_estimators: integer
        Number of trees in forest

        max_leaf_nodes: integer
        Maximum number of final nodes (less than number of data points) 

        max_features: integer
        Number of features to compare at each node in Tree

        cols: list
        List of features to select from DF to train/test Random Forest

        trainDF: DataFrame pandas
        Predetermined training set

        testDF: DataFrame pandas
        Predetermined test set
       

        Output
        ----------
        self:function
         - feature_importances_ = feature importances from random forest
         - preds = labels determined from random forest (test set)
         - reallabels = labels from training set
         - pred_proba = Probability of label (from random forest)
         - allPredicitions = Labels after defining random forest test on ALL data
         - allPredicitions_prob = Label probablility after defining random forest test on ALL data

        """

        if cols == None:
            #cols = ['g','m20','mprime','i','d','a','c','gr_col','f_gm20','logMass','ssfr']
            cols = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7', \
            'g','m20','mprime','i','d','a','c','gr_col','logMass','ssfr','f_gm20','d_gm20'] #,'mergerFlag']

        
        if trainDF==None or testDF==None:
            trainDF, testDF = trainingSet(df,training_fraction=trainfrac)
            print(len(trainDF),len(testDF),len(df))

        train = trainDF[cols].values
        test = testDF[cols].values
        labels = trainDF[traininglabel]

        clrf = RandomForestClassifier(n_jobs=-1,oob_score=True,class_weight='balanced_subsample',**kwargs)
        #Create random forest instance #

        clrf.fit(train,labels) #Train using visual classification training set
            
        preds = array(clrf.predict(test)) #Predict classifications for test set
        pred_proba = array(clrf.predict_proba(test)) #Predict classification probabilities for test set
        feature_importances_ = clrf.feature_importances_ #Importance of each feature
        
        self.feature_importances_ = feature_importances_
        self.preds = preds.astype(int)
        self.reallabels = testDF[traininglabel]
        self.pred_proba = pred_proba

        self.allPredicitions = array(clrf.predict(df[cols]))
        self.allPredicitions_prob = array(clrf.predict_proba(df[cols]))

        self.clrf=clrf

        self.testdf = testDF
        
        return

def randomForestMC(df,iterations=1000, thresh=0.4, n_estimators=1000,max_leaf_nodes=100,max_features=3,cols=None,\
                   trainDF=None,testDF=None, traininglabel='mergerFlag',trainfrac=0.67,**kwargs):
    """
    Parameters
    ----------
    df: DataFrame pandas
    Basis catalog for which to create Random Forest

    iterations: integer
    Number of times to create a random forest instance

    thresh: float
    Threshold of Predicion Probability used to classify

    n_estimators: integer
    Number of trees in forest

    max_leaf_nodes: integer
    Maximum number of final nodes (less than number of data points) 

    max_features: integer
    Number of features to compare at each node in Tree

    cols: list
    List of features to select from DF to train/test Random Forest

    trainDF: DataFrame pandas
    Predetermined training set

    testDF: DataFrame pandas
    Predetermined test set
   

    Output
    ----------
    result: Dataframe
    Feature importance and summary statistics determined for every iteration

    predicitions: array
    Labels after defining random forest test on ALL data

    predictions_proba: array
    Label probablility after defining random forest test on ALL data

    """

    d = {}
    rf_mc_df = pd.DataFrame(d)

    if cols == None:
        cols = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7', \
        'g','m20','mprime','i','d','a','c','gr_col','logMass','ssfr','f_gm20','d_gm20']

    import numpy as np
    summaryStatsIters = {'completeness': np.zeros(iterations), 'specificity': np.zeros(iterations),'risk':np.zeros(iterations),\
    'totalError':np.zeros(iterations),'ppv':np.zeros(iterations),'npv':np.zeros(iterations),'tpr':np.zeros(iterations),'fpr':np.zeros(iterations)}
    sumStatsItersDF = pd.DataFrame(summaryStatsIters)

    predictions = zeros((shape(df)[0],iterations))
    predictions_df = pd.DataFrame(predictions)
    predictions_proba = zeros((shape(df)[0],iterations))
    predictions_proba_df = pd.DataFrame(predictions_proba)

    dd = {}
    for colnames in cols:
        dd[colnames] = np.zeros(iterations)

    rf_mc_df = pd.DataFrame(dd)

    import timeit

    start = timeit.default_timer()

    rf_objs=[]

    Nroc=20
    threshes= np.logspace(-3.0,0,Nroc)
    ROCstats = {'thresh':np.zeros((Nroc,iterations)), 'tpr':np.zeros((Nroc,iterations)),'fpr':np.zeros((Nroc,iterations))}
    ROCtests = {'thresh':np.zeros((Nroc,iterations)), 'tpr':np.zeros((Nroc,iterations)),'fpr':np.zeros((Nroc,iterations))}
    
    for i in range(iterations): #Begin niterations number of Random forest
        print(str(i+1)+'/'+str(iterations))
        rf_mc = randomForest(df,cols=cols,n_estimators=n_estimators,max_leaf_nodes=max_leaf_nodes,max_features=max_features,\
                             trainDF=trainDF,testDF=testDF, traininglabel=traininglabel,trainfrac=trainfrac,**kwargs)
        predictions_df[i] = rf_mc.allPredicitions #Labels
        predictions_proba_df[i] = rf_mc.allPredicitions_prob[:,1] #Label Probability
        for colImportance in range(len(cols)): #Populate array of feature importances
            rf_mc_df[cols[colImportance]][i] = rf_mc.feature_importances_[colImportance]
        summaryStats = confusionMatrix(df,rf_mc.allPredicitions_prob,threshold=thresh,traininglabel=traininglabel) #Calculate summary statistics
        summaryStatsTest = confusionMatrix(rf_mc.testdf,rf_mc.pred_proba,threshold=thresh,traininglabel=traininglabel)
        
        for colStats in sumStatsItersDF.columns: #Add summary statistics into dataframe containing all iterations
            sumStatsItersDF[colStats][i] = summaryStatsTest[colStats]
        rf_objs.append(rf_mc)
        
        for j,tv in enumerate(threshes):
            ROC_results=confusionMatrix(df,rf_mc.allPredicitions_prob,threshold=tv,traininglabel=traininglabel)  #compute ROC curve
            ROC_tests=confusionMatrix(rf_mc.testdf,rf_mc.pred_proba,threshold=tv,traininglabel=traininglabel)
            for rs in ROCstats.keys():
                ROCstats[rs][j][i]=ROC_results[rs]
                ROCtests[rs][j][i]=ROC_tests[rs]
    
    stop = timeit.default_timer()

    print("Random Forest took ", stop - start, " seconds")

    result = pd.concat([rf_mc_df,sumStatsItersDF],axis=1,join_axes=[sumStatsItersDF.index])
    #Concatenate data frame with feature importances and summary statistics for every iteration of random forest

    return result, predictions_df, predictions_proba_df, rf_objs, ROCstats, ROCtests
