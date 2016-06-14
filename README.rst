PyML
=======================

A machine learning set of tools for using PCA to classify galaxies using agglomerative clustering and convex hulls.

Usage:

If catalog is a dictionary with the following parameters: ['C','M20','GINI','ASYM','MPRIME','I','D'], then 
use these steps to project morphological data onto predefined PC eigenvectors and to classify galaxies based
on the groups defined in Peth et al. 2015:
::

	from PyML import machinelearning as pyml
	from PyML import convexhullclassifier as cvx

	parameters = ['C_J','M20_J','GINI_J','ASYM_J','MPRIME_J','I_J','D_J'] 
	#Statistics MUST be in this order
	npmorph = pyml.dataMatrix(catalog,parameters) 
	pc = pyml.pcV(npmorph) #Principal Components
	groups  = cvx.convexHullClass(pc.X)	#Groups using convex hull classifier

For Random Forest Classifications, usage:
::
	from PyML import machinelearning as pyml
	import pandas as pd

	cols = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7',\
	'g','m20','mprime','i','d','a','c','gr_col','logMass','ssfr','f_gm20','d_gm20']

	#Use pandas to read in data as a dataframe (df)
	#df = pd.read_csv('DataFile.txt')

	result, labels, label_probability = ml.randomForestMC(df,iterations=1000)
	#result = summary statistics, feature importances (N iterations x N statistics/importances)
	#labels = labels following random forest (N galaxies x N iterations)
	#label_probability = probability of label following random forest (N galaxies x N iterations)


