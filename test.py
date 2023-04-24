import numpy as np
from sklearn.preprocessing import StandardScaler

my_array = np.arange(1, 2*3*10+1).reshape((2, 3, 10))
shapeor =my_array.shape
print("Original array:\n{0}\n".format(my_array))
print("Reshaped array:\n{0}\n".format(my_array.reshape(-1, my_array.shape[-1])))

my_array = my_array.reshape(-1, my_array.shape[-1])
print("Original array:\n{0}\n".format(my_array))
my_array = StandardScaler().fit_transform(my_array)
print("Original array:\n{0}\n".format(my_array))