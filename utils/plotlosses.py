import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab

log = logging.getLogger(__name__)

def plot_losses(epochs, losses, labels):

  fig_ = plt.figure(figsize=(32, 16))

  for i in range(len(losses)):

    plt.plot(epochs, losses[i], label=labels[i])

  plt.grid()

  plt.legend(loc=2, ncol=1)
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.show()