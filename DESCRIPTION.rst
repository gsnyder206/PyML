PyML
=======================

from PyML import machine as ml
from PyML import convexhullclassifier as cvx

PCA usage:
pc = ml.PCA(data) #if you want to calculate eigenvectors
pc = ml.pcV(data) #If you want to use eigenvectors calculated for 1.4 < z < 2 galaxies

Convex Hull usuage:
groups = cvx.convexHullClassifier(pc.X)

Random Forest usage: