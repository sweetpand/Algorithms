#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

__author__ = 'maxim'

import sys
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


class NeuralNetwork:
  class __Layer:
    def __init__(self, args):
      self.__epsilon = 5e-5
      self.localGrad = 0
      self.__weights = np.random.randn(args["previousLayerHeight"], args["height"]) * 0.01
      self.__biases = np.zeros((args["biasHeight"], 1))

    def __str__(self):
      return str(self.__weights)

    def forward(self, X):
      a = np.dot(X, self.__weights) + self.__biases
      self.localGrad = np.dot(X.T, self.__tanhPrime(a))
      return self.__tanh(a)

    def adjustWeights(self, err):
      self.__weights -= (err * self.__epsilon)

    def __tanh(self, z):
      return np.tanh(z)

    def __tanhPrime(self, a):
      return 1 - self.__tanh(a) ** 2

  def __init__(self, args):
    self.__inputDimensions = args["inputDimensions"]
    self.__outputDimensions = args["outputDimensions"]
    self.__hiddenDimensions = args["hiddenDimensions"]
    self.__layers = []
    self.__constructLayers()

  def __constructLayers(self):
    self.__layers.append(
      self.__Layer(
        {
          "biasHeight": self.__inputDimensions[0],
          "previousLayerHeight": self.__inputDimensions[1],
          "height": self.__hiddenDimensions[0][0] if len(self.__hiddenDimensions) > 0 else self.__outputDimensions[0]
        }
      )
    )

    for i in range(len(self.__hiddenDimensions)):
      self.__layers.append(
        self.__Layer(
          {
            "biasHeight": self.__hiddenDimensions[i + 1][0] if i + 1 < len(self.__hiddenDimensions) else self.__outputDimensions[0],
            "previousLayerHeight": self.__hiddenDimensions[i][0],
            "height": self.__hiddenDimensions[i + 1][0] if i + 1 < len(self.__hiddenDimensions) else self.__outputDimensions[0]
          }
        )
      )

  def forward(self, X):
    out = self.__layers[0].forward(X)
    for i in range(len(self.__layers) - 1):
      out = self.__layers[i + 1].forward(out)
    return out

  def train(self, X, Y, loss, epoch=100000):
    for i in range(epoch):
      YHat = self.forward(X)
      delta = -(Y - YHat)
      loss.append(sum(Y - YHat))
      err = np.sum(np.dot(self.__layers[-1].localGrad, delta.T), axis=1)
      err.shape = (self.__hiddenDimensions[-1][0], 1)
      self.__layers[-1].adjustWeights(err)
      i = 0
      for l in reversed(self.__layers[:-1]):
        err = np.dot(l.localGrad, err)
        l.adjustWeights(err)
        i += 1

  def printLayers(self):
    print("Layers:\n")
    for l in self.__layers:
      print(l)
      print("\n")


def main(args):
  X = np.array([[x, y] for x, y in product([0, 1], repeat=2)])
  Y = np.array([[0], [1], [1], [1]])
  nn = NeuralNetwork(
    {
      # (height,width)
      "inputDimensions": (4, 2),
      "outputDimensions": (1, 1),
      "hiddenDimensions": [
        (6, 1)
      ]
    }
  )

  print("input:\n\n", X, "\n")
  print("expected output:\n\n", Y, "\n")
  nn.printLayers()
  print("prior to training:\n\n", nn.forward(X), "\n")
  loss = []
  nn.train(X, Y, loss)
  print("post training:\n\n", nn.forward(X), "\n")
  nn.printLayers()
  fig, ax = plt.subplots()

  x = np.array([x for x in range(100000)])
  loss = np.array(loss)
  ax.plot(x, loss)
  ax.set(xlabel="epoch", ylabel="loss", title="logic gate training")

  plt.show()


if __name__ == "__main__":
  main(sys.argv[1:])
