import numpy as np
import matplotlib.pyplot as plt
import pickle
centroid = pickle.load(open('c_list.pkl', 'rb'))
num_points = pickle.load(open('pts_list.pkl', 'rb'))
print(len(num_points))
print(len(centroid))
rg = np.square(centroid[:, :2])
rg = np.sqrt(np.sum(rg, axis=1))
plt.scatter(rg, num_points, s=1)
plt.show()