"""
Vantage-point tree implementation. 
Please see the toturial: http://stevehanov.ca/blog/index.php?id=130 for reference.
"""
from collections import namedtuple
from collections import deque
import random
import numpy as np
import heapq
import time
from spatialtree import spatialtree
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats


class NDPoint(object):
    """
    A point in n-dimensional space
    """

    def __init__(self, x, idx=None):
        self.x = np.array(x)
        self.idx = idx
    def __repr__(self):
        return "NDPoint(idx=%s, x=%s)" % (self.idx, self.x)

class VPTree(object):
    """
    An efficient data structure to perform nearest-neighbor
    search. 
    """

    def __init__(self, points, dist_fn=None):
        self.left = None
        self.right = None
        self.mu = None
        self.dist_fn = dist_fn if dist_fn is not None else l2
        
        # choose a better vantage point selection process
        self.vp = points.pop(random.randrange(len(points)))

        if len(points) < 1:
            return

        # choose division boundary at median of distances
        distances = [self.dist_fn(self.vp, p) for p in points]
        self.mu = np.median(distances)
        left_points = []  # all points inside mu radius
        right_points = []  # all points outside mu radius
        for i, p in enumerate(points):
            d = distances[i]
            if d >= self.mu:
                right_points.append(p)
            else:
                left_points.append(p)

        if len(left_points) > 0:
            self.left = VPTree(points=left_points, dist_fn=self.dist_fn)

        if len(right_points) > 0:
            self.right = VPTree(points=right_points, dist_fn=self.dist_fn)

    def is_leaf(self):
        return (self.left is None) and (self.right is None)

class PriorityQueue(object):
    def __init__(self, size=None):
        self.queue = []
        self.size = size

    def push(self, priority, item):
        self.queue.append((priority, item))
        self.queue.sort()
        if self.size is not None and len(self.queue) > self.size:
            self.queue.pop()


# ## Distance functions
def l2(p1, p2):
    return np.sqrt(np.sum(np.power(p2.x - p1.x, 2)))

def l1(p1, p2):
    return np.sum(np.abs(p2.x - p1.x))

def get_nearest_neighbors(tree, q, k=1):
    """
    find k nearest neighbor(s) of q

    :param tree:  vp-tree
    :param q: a query point
    :param k: number of nearest neighbors

    """
    # buffer for nearest neightbors
    neighbors = PriorityQueue(k)
    get_nearest_neighbors.tau = np.inf
    #get_nearest_neighbors.count = 0
    def search(node, q, k, neighbors):
        if node is None:
            return
        d = node.dist_fn(q, node.vp)
        #get_nearest_neighbors.count += 1
        if(d < get_nearest_neighbors.tau):
            neighbors.push(d, node.vp)
            get_nearest_neighbors.tau, _ = neighbors.queue[-1]  # the biggest
        
        if node.is_leaf():
            return
        
        if d < node.mu:
            if d <= node.mu + get_nearest_neighbors.tau:
                search(node.left, q, k, neighbors)
                
            if d >= node.mu - get_nearest_neighbors.tau:
                search(node.right, q, k, neighbors)
        else:
            if d >= node.mu - get_nearest_neighbors.tau:
                search(node.right, q, k, neighbors)
            if d <= node.mu + get_nearest_neighbors.tau:
                search(node.left, q, k, neighbors)
    search(tree, q, k, neighbors)
    #print "search count:", get_nearest_neighbors.count
    return neighbors.queue

def get_all_in_range(tree, q, tau):
     """
    #     find all points within a given radius of point q
    # 
    #     :param tree: vp-tree
    #     :param q: a query point
    #     :param tau: the maximum distance from point q

    """
     neighbors = []
     #get_nearest_neighbors.count = 0
     def search1(node, q, neighbors, tau):
        if node is None:
            return
        d = node.dist_fn(q, node.vp)
        #get_nearest_neighbors.count += 1
        if(d < tau):
            neighbors.append((d, node.vp))
        
        if node.is_leaf():
            return
        
        if d < node.mu:
            if d <= node.mu + tau:
                search1(node.left, q, neighbors, tau)
                
            if d >= node.mu - tau:
                search1(node.right, q, neighbors, tau)
        else:
            if d >= node.mu - tau:
                search1(node.right, q, neighbors, tau)
            if d <= node.mu + tau:
                search1(node.left, q, neighbors, tau)
     search1(tree, q, neighbors, tau)
     #print "search count:", get_nearest_neighbors.count
     return neighbors


if __name__ == '__main__':
    X = np.random.uniform(0, 1, size=(1000,100))
    points = [NDPoint(x, i) for i, x in  enumerate(X)]
    tree = VPTree(points)
    result = []
    # query
    for item in range(0, 2):
        q = NDPoint(X[item], item)
        neighbors = get_nearest_neighbors(tree, q, k=21) 
        neighbors = get_all_in_range(tree, q, 3)
	print neighbors
