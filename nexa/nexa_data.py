"""
This is a file for the NexaData class. The NexaData class
should wrap the data that is going to be analyzed with Nexa.
"""


class NexaData():
    """
    This class is used to wrap the data.

    """
    def __init__(self, data_matrix, distance_matrix):
        """

        :param data_matrix: This should be a collection of all the sensors. A matrix
         where of dimensions Nsensors x Ndata
        :param distance_matrix: This should be a function that
        :return:
        """
        self.temporal_distance = distance_matrix
        self.DM =  data_matrix

    def calculate_data_matrix(self):
        """
        :return: data matrix, a matrix with Nsensors x Ndata dimensions.
        """

        return self.DM

    def calculate_sensor_distance_matrix(self):
        """
        Calculate the sensor distance for this matrix.

        :return: A matrix with Nsensors x Nsensors dimensions
        """

        return self.temporal_distance
