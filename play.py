import numpy as np
from nexa.nexa import Nexa, NexaData

# Load the file from the same directory
file = './percept_mat.csv'
with open(file, 'r') as f:
    data = np.loadtxt(f, dtype='float')

# We need to wrap the data in the Nexa format. For that we use NexaData.
data_matrix = data.T
# Let's add some noise to get rid of the zeros
data_matrix += np.random.rand(data_matrix.shape[0], data_matrix.shape[1])
distance_matrix = 1 - np.abs(np.corrcoef(data_matrix))
print('Zeros on the distance matrix', np.sum(distance_matrix == 0))
# This could be another possible way
# distance_matrix = np.dot(data_matrix, data_matrix.T)

nexa_data = NexaData(data_matrix, distance_matrix)

# Now we need to put this in the Nexa class. Which does all the data processing.
# But first we require some parameters

Nsensor_clusters = 5
Ndata_clusters =  5
Nembedding = 3

# Create the Nexa object
nexa_object = Nexa(nexa_data, Nsensor_clusters, Ndata_clusters, Nembedding)

# Doing the calculations
nexa_object.calculate_distance_matrix()  # So the nexa_objecct knows the distance
# Do the embedding
nexa_object.calculate_embedding()

# Now we calculate sensor clustering
nexa_object.calculate_sensor_clustering()
# And after this we can extract a map from sensor to cluster
sensor_to_cluster_mappping = nexa_object.sensor_to_cluster_mapping
# Which we can visualize
# visualization_clustering = np.vstack(sensor_to_cluster_mappping, sensor_to_cluster_mappping, sensor_to_cluster_mappping)
# And this you can use imshow to plot.

# Now we can the cluster to sensors
nexa_object.calculate_cluster_to_sensors()
# This makes available a map from a sensor to all the clusters that belong to it.
cluster_to_sensor_mapping = nexa_object.cluster_to_sensor_mapping

# And then we do the data clustering
nexa_object.calculate_data_clustering()
# This gives us access to a map from each cluster to all of the centers of the data clusters
# that belong to that particular sensor cluster. And you can access it like this.
cluster_to_data_centers_mapping = nexa_object.cluster_to_data_centers_mapping

# And if you just want to calculat everything and then extract it without doing it
# step by step you do the following
nexa_object.calculate_all()


