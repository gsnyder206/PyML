#!/usr/bin/env python

"""
My own personal Machine Learning codes.

Includes PCA, Diffusion Mapping, Random Forest
"""

__author__ = "Michael Peth, Peter Freeman"
__copyright__ = "Copyright 2015"
__credits__ = ["Michael Peth", "Peter Freeman"]
__license__ = "GPL"
__version__ = "0.2.3"
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


def whiten(data, A_basis=False):
    '''
    Compute a Principal Component analysis p for a data set

    Parameters
    ----------
    data: matrix
    Input data (Nxk): N objects by k features, whitened (i.e. average subtracted and stddev scaled)


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
        with open('candels_whiten_j_avg_std.txt', 'rb') as handle:
            a_basis = pickle.loads(handle.read())
        mu = a_basis['mean']
        wvar = a_basis['std']

    whiten_data = zeros(shape(data))
    for p in range(len(mu)):
        whiten_data[:,p]  = (data[:,p] - mu[p])/wvar[p]
    return whiten_data

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

        A_pcv: Matrix
        Data (NxK) with Eigenvector solutions used to project data

        Returns
        -------
        Structure with the following keys:

        X: matrix
        Principal Component Coordinates
        '''
        npmorph_path=PyML.__path__[0]+os.path.sep+"data"+os.path.sep+"npmorph_f125w_candels.txt" 
        with open(npmorph_path, 'rb') as handle:
            A_pcv = pickle.loads(handle.read())
        
        whiten_data = whiten(data) #,A_basis=A_pcv)
        A_white = whiten(A_pcv)
        pc1 = PCA(A_white)
        pc = zeros(shape(whiten_data))
        
        for i in range(len(whiten_data[0])):
             for j in range(len(whiten_data[0])):
                pc[:,i] = pc[:,i] + pc1.vectors[i][j]*whiten_data[:,j]

        self.X = pc
        self.vectors = pc1.vectors 
        self.values = pc1.values
        return

class pcVFix:
    def __init__(self,data):
        '''
        Compute a Principal Component analysis p for a data set

        Parameters
        ----------
        data: matrix
        Input data (Nxk): N objects by k features

        A_pcv: Matrix
        Data (NxK) with Eigenvector solutions used to project data

        Returns
        -------
        Structure with the following keys:

        X: matrix
        Principal Component Coordinates
        '''
        npmorph_path="PC_f125w_candels.txt" 
        with open(npmorph_path, 'rb') as handle:
            pc1 = pickle.loads(handle.read())
        
        whiten_data = whiten(data,A_basis=True)
        pc = zeros(shape(whiten_data))
        
        for i in range(len(whiten_data[0])):
             for j in range(len(whiten_data[0])):
                pc[:,i] = pc[:,i] + pc1['vectors'][i][j]*whiten_data[:,j]

        self.X = pc
        self.vectors = pc1['vectors']
        self.values = pc1['values']
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

def morphologySelection(catalog,parameters):

    A = zeros((len(catalog['RA_J']),len(parameters)))

    npar = 0
    for param in parameters:
        #print param
        A[:,npar] = catalog[param]
        npar+=1

    return A
        

class randomForest:
    def __init__(self, catalog,train,test=None,niter=1000,classify='mergers'):

        """
        Parameters
        ----------
        catalog: Dictionary
        Basis catalog for which to create Random Forest
        
        data:Matrix [nsample,nfeatures]

        classify: String
        Determines the type of classification forest to make

        Output
        ----------
        labels:Array [nsample]
        - Labels corresponding to classifications you wish to make (i.e. 0=non-merger, 1=merger)

        """

        if classify=='mergers':
            merger_idx = where((catalog['MERGER'] > 0.5))[0]
            print "%i Merging Galaxies" % len(merger_idx)
            labels = zeros(len(catalog['MERGER']))
            labels[merger_idx] = 1
            targetNames = array(['non-mergers','mergers'])

        elif classify=='irr':
            irr_idx = where((catalog['IRR']  >= 2/3.))[0]
            print "%i Irregular Galaxies" % len(irr_idx)
            labels = zeros(len(catalog['IRR']))
            labels[irr_idx] = 1

        elif classify=='sph':
            sph_idx = where((catalog['SPHEROID']  >= 2/3.))[0]
            print "%i Pure Spheroidal Galaxies" % len(sph_idx)
            labels = zeros(len(catalog['SPHEROID']))
            labels[sph_idx] = 1

        elif classify=='disk':
            sph_idx = where((catalog['DISK']  >= 2/3.))[0]
            print "%i Pure Disk Galaxies" % len(sph_idx)
            labels = zeros(len(catalog['DISK']))
            labels[sph_idx] = 1

        elif classify=='morph':
            d_idx   = where((catalog['disk'.upper()] >= 2/3.) & (catalog['spheroid'.upper()]  < 2/3.))[0]
            sph_idx = where((catalog['spheroid'.upper()]  >= 2/3.) & (catalog['disk'.upper()]  < 2/3.))[0]
            irr_idx = where((catalog['irr'.upper()]  >= 2/3.) & (catalog['disk'.upper()]  < 2/3.) & (catalog['spheroid'.upper()] < 2/3.))[0]
            dirr_idx = where((catalog['irr'.upper()]  >= 2/3.) & (catalog['disk'.upper()]  >= 2/3.) & (catalog['spheroid'.upper()] < 2/3.))[0]
            ds_idx  = where((catalog['disk'.upper()]  >= 2/3.) & (catalog['spheroid'.upper()]  >= 2/3.))[0]

            labels = zeros(len(catalog['DISK']))
            labels[d_idx] = 1
            labels[sph_idx] = 2
            labels[irr_idx] = 3
            labels[ds_idx] = 4
            labels[dirr_idx] = 5
            targetNames = array(['other','disk','spheroid','irregular','disk+sph','irr-disk'])
            
        elif classify=='clumpy':
            clumpy_class = where((catalog['c1p0'.upper()] > 0.5) | (catalog['c2p0'.upper()] > 0.5) | (catalog['c1p1'.upper()] > 0.5) | \
                            (catalog['c1p2'.upper()] > 0.5) | (catalog['c2p1'.upper()] > 0.5) | (catalog['c2p2'.upper()] > 0.5))[0]
            print "%i Clumpy Galaxies" % len(clumpy_class)            
            labels = zeros(len(catalog['MERGER']))
            labels[clumpy_class] = 1

        else:
            pass


        clf = RandomForestClassifier(n_jobs=2,n_estimators=500) #Create random forest instance
        clf.fit(train,labels) #Train using visual classification training set
        preds = array(clf.predict(test)) #Predict classifications for test set
        pred_proba = array(clf.predict_proba(test)) #Predict classification probabilities for test set
        feature_importances_ = clf.feature_importances_ #Importance of each feature
        
        self.feature_importances_ = feature_importances_
        self.preds = preds.astype(int)
        self.pred_proba = pred_proba
        return
            
        
def morphTrainTest(catalog,classify='morph'):

    if classify=='mergers':
        merger_idx = where((catalog['MERGER'] > 0.5))[0]
        print "%i Merging Galaxies" % len(merger_idx)
        labels = zeros(len(catalog['MERGER']))
        labels[merger_idx] = 1
        targetNames = array(['non-mergers','mergers'])
    elif classify=='irr':
        irr_idx = where((catalog['IRR']  >= 2/3.))[0]
        print "%i Irregular Galaxies" % len(irr_idx)
        labels = zeros(len(catalog['IRR']))
        labels[irr_idx] = 1

    elif classify=='sph':
        sph_idx = where((catalog['SPHEROID']  >= 2/3.))[0]
        print "%i Pure Spheroidal Galaxies" % len(sph_idx)
        labels = zeros(len(catalog['SPHEROID']))
        labels[sph_idx] = 1

    elif classify=='disk':
        sph_idx = where((catalog['DISK']  >= 2/3.))[0]
        print "%i Pure Disk Galaxies" % len(sph_idx)
        labels = zeros(len(catalog['DISK']))
        labels[sph_idx] = 1

    elif classify=='morph':
        d_idx   = where((catalog['disk'.upper()] >= 2/3.) & (catalog['spheroid'.upper()]  < 2/3.))[0]
        sph_idx = where((catalog['spheroid'.upper()]  >= 2/3.) & (catalog['disk'.upper()]  < 2/3.))[0]
        irr_idx = where((catalog['irr'.upper()]  >= 2/3.) & (catalog['disk'.upper()]  < 2/3.) & (catalog['spheroid'.upper()] < 2/3.))[0]
        dirr_idx = where((catalog['irr'.upper()]  >= 2/3.) & (catalog['disk'.upper()]  >= 2/3.) & (catalog['spheroid'.upper()] < 2/3.))[0]
        ds_idx  = where((catalog['disk'.upper()]  >= 2/3.) & (catalog['spheroid'.upper()]  >= 2/3.))[0]
        labels = zeros(len(catalog['DISK']))
        labels[d_idx] = 1
        labels[sph_idx] = 2
        labels[irr_idx] = 3
        labels[ds_idx] = 4
        labels[dirr_idx] = 5
        targetNames = array(['other','disk','spheroid','irregular','disk+sph','irr-disk'])
            
    elif classify=='clumpy':
        clumpy_class = where((catalog['c1p0'.upper()] > 0.5) | (catalog['c2p0'.upper()] > 0.5) | (catalog['c1p1'.upper()] > 0.5) | \
                            (catalog['c1p2'.upper()] > 0.5) | (catalog['c2p1'.upper()] > 0.5) | (catalog['c2p2'.upper()] > 0.5))[0]
        print "%i Clumpy Galaxies" % len(clumpy_class)
        labels = zeros(len(catalog['MERGER']))
        labels[clumpy_class] = 1

    else:
        pass

    return labels