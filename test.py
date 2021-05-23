from itertools import count
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import random
import ujson
from plot_utils import plot_blackjack_values, plot_policy, plot_rewards


lst = []
nums = []

index = count()


# def animate(i):
#         lst.append(next(index))
#         nums.append(random.randint(0, 5))
    
#         plt.cla()
#         plt.plot(lst, nums)
#         plt.show()

# def plot():     
#     bruh = FuncAnimation(plt.gcf(), animate, interval=100)
#     plt.show()

    

# plot()

# file = open('q_values.json')
# q_values = ujson.load(file)
# file.close()
# print(q_values)
# print(len(q_values))


# test = count(start=1)
# for i in test:
#     print(i)
#     if i == 10:
#         break
lst = [[1, 2, 3, 4, 5, 6, 7]]
plot_rewards(lst) 
