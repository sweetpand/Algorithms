#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'maxim'

import numpy as np


def kmeans_assignment(centroids, points):
  num_centroids, dim = centroids.shape
  num_points, _ = points.shape

  # Reshape both arrays into `[num_points, num_centroids, dim]`
  centroids = np.tile(centroids, [num_points, 1]).reshape([num_points, num_centroids, dim])
  points = np.tile(points, [1, num_centroids]).reshape([num_points, num_centroids, dim])

  # Compute all distances (for all points and all centroids) at once and select the min centroid for each point
  distances = np.sum(np.square(centroids - points), axis=2)
  return np.argmin(distances, axis=1)


def main():
  centroids = np.array([
    [1, 2, 1, 1],
    [4, 2, 0, -1],
    [3, 1, 1, 4],
  ])

  points = np.array([
    [1, 0, 1, 1],
    [4, 1, 1, 1],
    [3, 1, 1, 1],
    [2, 0, 1, 3],
    [4, 2, 0, 0],
  ])

  centroid_group = kmeans_assignment(centroids, points)
  print(centroid_group)


if __name__ == '__main__':
  main()
