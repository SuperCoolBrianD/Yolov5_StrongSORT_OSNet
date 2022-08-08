import numpy
import pickle
import math
import matplotlib.pyplot as plt
import numpy as np
mes = pickle.load(open('extrinsic_meas.pkl', 'rb'))

print(mes)


# d1 = [19.707622243928856, -4.7934115407457885, 36.488593042009995, 36.802095271660484]
# d2 = [0.3368208135438708, -3.991029010989413, 42.36073352693984, 42.548326142246346]
# # d2 = [6.389254165876508, 1.3340224749974408, 29.954029230683478, 30.656908714229633]



#
# tana = (d1[2] - d2[2])/(d2[1]-d1[1])
# alpha = abs(math.atan(tana))
# print(math.degrees(math.atan(alpha)))
# theta = math.sin(d2[1]/d2[3])
# h = d2[3] * math.cos(alpha+theta)
# print(h)

h_list = []
d_list = []
for i, d1 in enumerate(mes):
    for d2 in mes[i+1:]:
        tana = (d1[2] - d2[2]) / (d2[1] - d1[1])
        alpha = abs(math.atan(tana))
        theta = math.sin(d2[1] / d2[3])
        if alpha > np.pi/4 and alpha-theta < np.pi/2:
            h = d2[3] * math.cos(alpha - theta)
            d_list.append(alpha)
            h_list.append(h)

# h_list = []
# alpha = 1.2
# for d1 in mes:
#     theta = math.sin(d1[1] / d1[3])
#     h = d1[3] * math.cos(alpha - theta)
#     h_list.append(h)
n, bins, patches = plt.hist(d_list, 200, facecolor='g', alpha=0.75)
plt.title('Radar Height Histogram')
plt.xlabel("Height (m)")
plt.show()