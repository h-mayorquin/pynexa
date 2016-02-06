import numpy as np
from MNIST.load_data import training_ims, training_labels

Ndata = 1000
X = training_ims[:Ndata]
Y = training_labels[:Ndata]