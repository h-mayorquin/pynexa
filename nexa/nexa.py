"""
This is the main class for Nexa.
"""

import numpy as np
from sklearn import manifold, cluster

from nexa.nexa_data import NexaData

class Nexa():
    """
    This is the main class for Nexa.
    """

    n_init = 10   # This is for initialization of the sklearn algorithms
    n_jobs = -1   # Use all cpus, 1 for only one.

    def __init__(self, nexa_data, Nsensor_clusters, Ndata_clusters, Nembedding):
        """

        :param nexa_data: The sensors wrapped in a nexa_data object.
        :param Nsensor_clusters: The number of clusters for the sensors.
        :param Ndata_clusters: The number of clusters for the data.
        :param Nembedding: The dimension of the space to embedd within.

        :return:  None
        """

        self.Nsensor_clusters = Nsensor_clusters
        self.Ndata_clusters = Ndata_clusters
        self.Nembedding = Nembedding

        # Check that  nexa data is an instance of NexaData
        if isinstance(nexa_data, NexaData):
            self.nexa_data = nexa_data
        else:
           raise TypeError('Data should come wrapped in a NexaObject')

        self.data_matrix = nexa_data.calculate_data_matrix()
        self.Nsensors = self.data_matrix.shape[0]
        self.Ndata = self.data_matrix.shape[1]

        # Quantities to calculate
        self.sensor_distance_matrix = None
        self.embedding = None
        self.sensor_to_cluster_mapping = None
        self.cluster_to_sensor_mapping = None
        self.cluster_to_data_centers_mapping = None

    def calculate_distance_matirx(self):
        """
        Calculates the distance between the sensors.

        :return: Calculates similar distance matrix
        """

        self.sensor_distance_matrix = self.nexa_data.calculate_sensor_distance_matrix()
        return self.sensor_distance_matrix

    def calculate_embedding(self):
        """
        This calculates the euclidian embedding of our
        distance matrix using MDS.

        Calculates the embedding, that is, it embedds every
        sensor on an Euclidian space with dimensions equal to
        self.Nembedding. Threfore it should return an array
        with a shape = (self.Nsensors, self.Nembedding).

        :return: the stress, a measure of how fit the embedding is.
        """

        disimi = 'precomputed'
        n_init = Nexa.n_init
        n_jobs = Nexa.n_jobs
        n_comp = self.Nembedding

        classifier = manifold.MDS(n_components=n_comp, n_init=n_init,
                                  n_jobs=n_jobs, dissimilarity=disimi)

        self.embedding = classifier.fit_transform(self.sensor_distance_matrix)

        return classifier.stress_

    def calculate_sensor_clustering(self, centers=False):
        """
        This class calculates the spatial clustering. Once there is
        an embedding this function performs a clustering in the
        embedded space with as many clusters as self.Nsensors_clusters

        It returns an index to cluster which maps every sensor to the
        cluster (via a number) that it belongs to.

        :param centers: If true the function returns the centers of the
        sensor clustering.
        :return: centers of the sensor clustering.
        """
        n_jobs = Nexa.n_jobs
        n_clusters = self.Nsensor_clusters

        classifier = cluster.KMeans(n_clusters=n_clusters, n_jobs=n_jobs)
        self.sensor_to_cluster_mapping = classifier.fit_predict(self.embedding)

        if centers:
            return classifier.cluster_centers_

    def calculate_cluster_to_sensors(self):
        """
        Calculates the dictionary where each cluster maps
        to the set of all sensors that belong to it. It should
        therefore return a dictionary with as many elements as
        there a clusters each mapping to a subset of the sensor
        set.

        :return: None
        """
        self.cluster_to_sensor_mapping = {}

        for cluster_n in range(self.Nsensor_clusters):
            indexes = np.where(self.sensor_to_cluster_mapping == cluster_n)[0]
            self.cluster_to_sensor_mapping[cluster_n] = indexes

    def calculate_data_clustering(self):
        """
        This calculates a dictionary where the keys are sensor
        cluster indexes and the values are an array
        of the cluster centers.

        :return:
        """
        n_jobs = Nexa.n_jobs
        t_clusters = self.Ndata_clusters

        self.cluster_to_data_centers_mapping= {}

        for cluster_n, cluster_indexes in self.cluster_to_sensor_mapping.items():

            data_in_the_cluster = self.data_matrix[cluster_indexes, :]
            classifier = cluster.KMeans(n_clusters=t_clusters, n_jobs=n_jobs)
            classifier.fit_predict(data_in_the_cluster.T)
            centers = classifier.cluster_centers_
            self.cluster_to_data_centers_mapping[cluster_n] = centers

