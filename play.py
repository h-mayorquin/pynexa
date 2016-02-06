import numpy as np
from MNIST.load_data import training_ims, training_labels
from nexa.nexa import Nexa, NexaData

Ndata = 1000
X = training_ims[:Ndata]
Y = training_labels[:Ndata]

# Convert to float
X = X.astype('float')

# Add some noise
X += np.random.rand(X.shape[0], X.shape[1])

# Need to calculate the distance and the data in X
data = X.T
distance = 1 - np.abs(np.corrcoef(data))
nexa_data = NexaData(data, distance)

# Now we do the Nexa thing
Nsensor_clustering = 5
Ndata_clustering = 10
Nembedding = 3

# Calculate all the quantities
nexa_object = Nexa(nexa_data, Nsensor_clustering, Ndata_clustering, Nembedding)

# First we calculate the distance matrix
nexa_object.calculate_distance_matrix()

# Now we calculate the embedding
nexa_object.calculate_embedding()

# Now we calculat how the sensors clustering, this also makes
# available a mapping from each sensor to the cluster that it belongs
nexa_object.calculate_sensor_clustering()

# Get's a map from each to cluster to all the sensors wichh belong to it.
nexa_object.calculate_cluster_to_sensors()

# Calculates teh data clustering and also makes available a map from each
# cluster to its respective time centers.
nexa_object.calculate_data_clustering()

# Does all of the above.
nexa_object.calculate_all()
