PyML
=======================

A machine learning set of tools for using PCA to classify galaxies using agglomerative clustering and convex hulls.

Usage:

If catalog is a dictionary with the following parameters: ['C','M20','GINI','ASYM','MPRIME','I','D'], then 
use these steps to project morphological data onto predefined PC eigenvectors and to classify galaxies based
on the groups defined in Peth et al. 2015:

from PyML import machinelearning as pyml
from PyML import convexhullclassifier as cvx

npmorph = pyml.dataMatrix(catalog,['C_J','M20_J','GINI_J','ASYM_J','MPRIME_J','I_J','D_J']) #Statistics MUST be in this order
pc = pyml.pcV(npmorph)																		#Principal Components
groups  = cvx.convexHullClass(pc.X)															#Groups using convex hull classifier