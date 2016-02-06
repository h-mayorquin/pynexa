# pynexa
The nexa module in Python.

This is a module that recreates the Nexa data processing pipeline in Python (from its original C++)


There are two important classes one that wraps the data in a specific format and
another class that knows how to process that specfici format.

In the file `mnist_demo.py` an example on how to run the pipeline can be found.

If you run it you should produce a segmentation of the space given by the
sensor clustering obtained along the **Nexa** pipeline. The segmentation should 
look like this.

![mnist_example](https://github.com/h-mayorquin/pynexa/blob/master/mnist_sensor_clustering.png)
