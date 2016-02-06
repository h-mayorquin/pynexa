import numpy as np
from MNIST.load_data import training_ims, training_labels
from nexa.nexa import Nexa, NexaData

Ndata = 1000
X = training_ims[:Ndata]
Y = training_labels[:Ndata]

# Need to calculate the distance and the data in X