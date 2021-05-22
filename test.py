from itertools import count
import matplotlib.pyplot as plt
lst = [1000, 2000, 3000]

def animate():
    value = next(iter(lst))
    y_vals = next(count())
  
    
# animate()
# plt.show()

x = [1, 2, 3]
y = [5, 7, 8]
plt.scatter(x, y)
plt.show()