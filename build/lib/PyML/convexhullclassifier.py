#!/usr/bin/env python

"""
Classify galaxies using their PCs and comparing the results to a basis set of 
groups defined for F125W 1.4 < z < 2 galaxies in UDS+GOODS-S+COSMOS

Galaxies classified based on 2D convex hulls since using the full 7D convex hull is too slow
and impossible for groups with many points (aka Group 6)
"""

from numpy import *
#import pyfits
#from pygoods import *
from sklearn.cluster import AgglomerativeClustering 
import scipy
from scipy import spatial
from matplotlib import path
import pickle
import PyML
import os

__author__ = "Michael Peth"
__copyright__ = "Copyright 2015"
__credits__ = ["Michael Peth"]
__license__ = "GPL"
__version__ = "0.2.6"
__maintainer__ = "Michael Peth"
__email__ = "mikepeth@jhu.edu"

def clusterCentroids(X,labels):
    """
    Find the centeroids for each group following the PCA
    """
    m = []
    for i in range(10):
        grp = where(labels==i)[0]
        if len(grp) > 0:
            m += [median(X[grp,:],axis=0)]
        else:
            m += [zeros(7)+10000.]
    return m
        
def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimension
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimension for which a Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    #from pyhull.delaunay import DelaunayTri
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0

def convexHullClass(data,radius=1e-2):
    #from matplotlib import nxutils
    """
    Ward clustering based on a Feature matrix.

    Recursively merges the pair of clusters that minimally increases
    within-cluster variance.

    The inertia matrix uses a Heapq-based representation.

    This is the structured version, that takes into account some topological
    structure between samples.

    Parameters
    ----------
    data : array, shape (n_samples, n_features) = (N,7)
        feature matrix  representing n_samples samples to be clustered based on clusters found for X (PC data)

    radius : float (optional)
        The search radius for determining if a point is within a convex hull
        Radius of 1e-6 gives lowest number of mistaken labels

    Returns
    -------
    new_labels : 1D array, shape (n_samples)
        The cluster labels for data based upon clusters created for X
    """
    
    #Define groups using basis data from 1.4 < z < 2
    pc_path=PyML.__path__[0]+os.path.sep+"data"+os.path.sep+'pc_f125w_candels.txt'
    with open(pc_path, 'rb') as handle:
        pc_j_dict = pickle.loads(handle.read())
    
    #Select PCs from dictionary key 'X' and groups from key 'label'
    X = pc_j_dict['X']
    label = pc_j_dict['label']
    ev_weight = pc_j_dict['values']
    ev_weight_normalized = ev_weight[0]*ev_weight[0]
    clusters = 10
    #groupALL = []

    clusterCenters = clusterCentroids(X,label)
    
    group_label = zeros(int(len(data)))-1 #Array defining the new group labels
    distance_sq = zeros(clusters) #Array defining the distances to different groups, refreshed for every galaxy. Used in case of ties
    for ngal in range(len(data)):
        dist = zeros(clusters)
        if ((data[ngal,1] < -4) | (data[ngal,0] > 4.79)): #These definitions of "outliers" are hard coded depend on basis data
                group_label[ngal] = -1 #outliers are put into group =-1, replaces groups 3,7
                #print "Galaxy ", ngal, " is an outlier"
                #continue
                #groupALL.append(group_label)
        else: #If galaxy is not an "outlier" proceed to check which group it belongs
            for n in range(clusters): #len(unique(label))):
                #groupn = unique(label)[n]
                labeln = where(label == n)[0]
                dist[n] = scipy.spatial.distance.sqeuclidean(clusterCenters[n],data[ngal]) #find distance from data point to cluster centers

            #Distances to groups 3 and 7 are nan, so this fixes that
            dist[3] = 1000.0 
            dist[7] = 1000.0
            #print group
            groups2 = dist - min(dist) #subtract smallest distance value from all distances

            if min(dist) > 20:
                group_label[ngal] = -1 #outliers are put into group =-1, replaces groups 3,7
                print "Galaxy ", ngal, " is an outlier"
                #continue
                #groupALL.append(dist)
            else:
                multiple_groups_possible = where(groups2 < 8)[0] #If the new distance is smaller than some arbitrary value do convex hull test
                #print multiple_groups_possible

                if len(multiple_groups_possible) == 1:
                    #print "Only 1 Group Galaxy ", ngal
                    #group_number = [argmin(dist)]
                    group_label[ngal] = argmin(dist)
                    #groupALL.append(dist)
                else:       
                    group = zeros(clusters) 
                    for n in multiple_groups_possible: #Only look in convex hulls that are close to data point #range(clusters): #len(unique(label))):
                        labeln = where(label == n)[0]

                        #First test if the galaxy is even classified in PC1, if not skip to using closest galaxy group
                        for pcx in range(1):
                            for pcy in range(3): #shape(X)[1]):
                                if pcy > pcx: #Cycle through all iterations of PCx,PCy combinations to check if galaxy is in a convex hull
                                    if len(labeln) > 14: #For a convex hull to work properly there must be at least 14 data points to define the hul
                                        group1_pt_x,group1_pt_y  = data[ngal][pcx], data[ngal][pcy] #Grab data for galaxy at PCx and PCy
                                        points = zeros((len(labeln),2))
                                        points[:,0] = X[labeln][:,pcx]#Pick galaxies only for a specific group
                                        points[:,1] = X[labeln][:,pcy]
                                        hull = spatial.ConvexHull(points)

                                        inside_poly_shape = path.Path(points[hull.vertices]).contains_point((group1_pt_x,group1_pt_y),radius=radius)
                                        if inside_poly_shape ==True:
                                            group[n] = group[n]+ev_weight[pcx]*ev_weight[pcy]/ev_weight_normalized



                    if sum(group) == 0:
                        group_label[ngal] = argmin(dist) #If galaxy isn't grouped after PC1 then use closest group
                        #groupALL.append(dist)
                        #group[group_number] = 1e6
                        #print "Galaxy Breaks ", ngal
                        #continue
                        
                    else:
                        group = zeros(10)
                        for n in multiple_groups_possible: #Only look in convex hulls that are close to data point #range(clusters): #len(unique(label))):
                            labeln = where(label == n)[0]                        
                            for pcx in range(shape(X)[1]):
                                for pcy in range(shape(X)[1]):
                                    if pcy > pcx: #Cycle through all iterations of PCx,PCy combinations to check if galaxy is in a convex hull
                                        if len(labeln) > 14: #For a convex hull to work properly there must be at least 14 data points to define the hul
                                            group1_pt_x,group1_pt_y  = data[ngal][pcx], data[ngal][pcy] #Grab data for galaxy at PCx and PCy
                                            points = zeros((len(labeln),2))
                                            points[:,0] = X[labeln][:,pcx]#Pick galaxies only for a specific group
                                            points[:,1] = X[labeln][:,pcy]
                                            hull = spatial.ConvexHull(points)

                                            #Does Polygon (2D convex hull) contain point?
                                            inside_poly_shape = path.Path(points[hull.vertices]).contains_point((group1_pt_x,group1_pt_y),radius=radius)
                                            #inside_poly_shape = in_hull((group1_pt_x,group1_pt_y),points)
                                            if inside_poly_shape ==True:
                                                #if pcx == 0:
                                                #    group[n] = group[n]+1
                                                #else:
                                                group[n] = group[n]+ev_weight[pcx]*ev_weight[pcy]/ev_weight_normalized

                        #print "Does this happen?"                        
                        group_number = where(group == max(group))[0]
                    
                        if len(group_number) > 1: #In case there is a tie for the group most classified into
                            #print "Galaxy ", ngal
                            var_new = zeros(len(group_number))
                            
                            #Determine the distance to the "center" of all convex hulls and the closest is used to break ties
                            distance_sq = zeros(clusters)
                            for nn in range(clusters):
                                #eq_label = where(label == group_number[nn])[0]

                                # galaxy = data[ngal,:].reshape((1,len(data[0])))
                                # new_points = concatenate((X[eq_label],galaxy))
                                
                                # for sample_ngal in range(len(eq_label)):
                                #     distance_sq[nn] = distance_sq[nn]+scipy.spatial.distance.sqeuclidean(data[ngal],X[eq_label][sample_ngal])

                                distance_sq[nn] = scipy.spatial.distance.sqeuclidean(clusterCenters[nn],data[ngal])
                                #Distances to groups 3 and 7 are nan, so this fixes that
                                distance_sq[3] = 1000.0
                                distance_sq[7] = 1000.0

                            group_label[ngal] = argmin(distance_sq)
                            #groupALL.append(distance_sq)
                            # if len(where(distance_sq == min(distance_sq))[0]) == 1:
                            #     group_label[ngal] = group_number[where(distance_sq == min(distance_sq))[0]]
                            # else: #In the off chance the galaxy is equally close to two groups, labeled as group -2
                            #     "Galaxy ", ngal, " apparently is either really in multiple groups or none"
                            #     group_label[ngal] = -2

                        #if len(group_number) == 1:
                        #print "ayyyy lmao"
                        else:
                            #groupALL.append(group)
                            group_label[ngal] = group_number[0]
            
    return array(group_label,'i') #, groupALL

# def convexHullClass(data,radius=1e-6):
#     #from matplotlib import nxutils
#     """
#     Ward clustering based on a Feature matrix.

#     Recursively merges the pair of clusters that minimally increases
#     within-cluster variance.

#     The inertia matrix uses a Heapq-based representation.

#     This is the structured version, that takes into account some topological
#     structure between samples.

#     Parameters
#     ----------
#     data : array, shape (n_samples, n_features) = (N,7)
#         feature matrix  representing n_samples samples to be clustered based on clusters found for X

#     radius : float (optional)
#         The search radius for determining if a point is within a convex hull
#         Radius of 1e-6 gives lowest number of mistaken labels

#     Returns
#     -------
#     new_labels : 1D array, shape (n_samples)
#         The cluster labels for data based upon clusters created for X
#     """
    
#     #Define groups using basis data from 1.4 < z < 2
#     pc_path=PyML.__path__[0]+os.path.sep+"data"+os.path.sep+'pc_f125w_candels.txt'
#     with open(pc_path, 'rb') as handle:
#         pc_j_dict = pickle.loads(handle.read())
    
#     #Select PCs from dictionary key 'X' and groups from key 'label'
#     X = pc_j_dict['X']
#     label = pc_j_dict['label']
#     clusters = 10
    
#     group_label = zeros(int(len(data)))-1 #Array defining the new group labels
#     distance_sq = zeros(clusters) #Array defining the distances to different groups, refreshed for every galaxy. Used in case of ties
#     for ngal in range(len(data)):
#         group = zeros(clusters)
#         if ((data[ngal,1] < -4) | (data[ngal,0] > 4.79)): #These definitions of "outliers" are hard coded depend on basis data
#                 group_label[ngal] = -1 #outliers are put into group =-1, replaces groups 3,7
#                 #print "Galaxy ", ngal, " is an outlier"
#                 continue
#         else: #If galaxy is not an "outlier" proceed to check which group it belongs
#             for n in unique(label):
#                 labeln = where(label == n)[0]
                

#                 if len(labeln) > 14: #For a convex hull to work properly there must be at least 14 data points to define the hull
#                     for pcx in range(shape(X)[1]):
#                         for pcy in range(shape(X)[1]):
#                             if pcy > pcx: #Cycle through all iterations of PCx,PCy combinations to check if galaxy is in a convex hull
#                                 group1_pt_x,group1_pt_y  = data[ngal][pcx], data[ngal][pcy] #Grab data for galaxy at PCx and PCy
#                                 points = zeros((len(labeln),2))
#                                 points[:,0] = X[labeln][:,pcx]#Pick galaxies only for a specific group
#                                 points[:,1] = X[labeln][:,pcy]
#                                 hull = spatial.ConvexHull(points)

#                                 #Does Polygon (2D convex hull) contain point?
#                                 inside_poly_shape = path.Path(points[hull.vertices]).contains_point((group1_pt_x,group1_pt_y),radius=radius)
#                                 #inside_poly_shape = in_hull((group1_pt_x,group1_pt_y),points)
#                                 if inside_poly_shape ==True:
#                                     group[n] = group[n]+1
#                 else:
#                     pass

#             group_number = where(group == max(group))[0] #After cycling through PCx iterations the group the galaxy is classified into most is used
            
#             if len(group_number) > 1: #In case there is a tie for the group most classified into
#                 var_new = zeros(len(group_number))
                
#                 #Determine the distance to the "center" of all convex hulls and the closest is used to break ties
#                 distance_sq = zeros(len(group_number))
#                 for nn in range(len(group_number)):
#                     eq_label = where(label == group_number[nn])[0]

#                     galaxy = data[ngal,:].reshape((1,len(data[0])))
#                     new_points = concatenate((X[eq_label],galaxy))
      
#                     for sample_ngal in range(len(eq_label)):
#                         distance_sq[nn] = distance_sq[nn]+scipy.spatial.distance.sqeuclidean(data[ngal],X[eq_label][sample_ngal])

#                 if len(where(distance_sq == min(distance_sq))[0]) == 1:
#                     group_label[ngal] = group_number[where(distance_sq == min(distance_sq))[0]]
#                 else: #In the off chance the galaxy is equally close to two groups, labeled as group -2
#                     "Galaxy ", ngal, " apparently is either really in multiple groups or none"
#                     group_label[ngal] = -2

#             if len(group_number) == 1:
#                 group_label[ngal] = group_number[0]
    
#     return array(group_label,'i')
