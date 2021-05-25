import json 

file = open('misc/q_values.json')
Q = json.load(file)
file.close()
print(Q)

