import pickle
import numpy as np
with open('test6.pkl', 'rb') as file:
    data = pickle.load(file)

l = len(data)

arr = np.zeros((l, 20), dtype=np.str)
for row, i in enumerate(data):
    # print(row)
    print(i)
    for col, j in enumerate(i):
        # print(col)
        # print(row)
        s = ' '.join([str(item) for item in j])
        arr[row, col] = s
# print(arr)

