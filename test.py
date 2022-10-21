from queue import Queue


q = Queue(maxsize=20)
for i in range(10):
    q.put(i)

while not q.empty():
    print(q.get())