from __future__ import division
import numpy as np
import os
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

path = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(path, './ml-bugs.csv'))
# X = data.iloc[:,1:]
# y = data.iloc[:,0]

# Xp = list(map(lambda x: [x[0], x[1], float(x[1]) > 17, float(x[1]) > 20], X.values))

def grouper(vect):
  grouped = {}

  for v in vect:
    if not grouped.get(v):
      grouped[v] = 0
    grouped[v]+= 1

  return grouped

def getEntropy(vect):
  sum = 0
  # total = vect.shape[0]
  total = len(vect)
  grouped = grouper(vect)
  print(grouped)

  for g in grouped:
    sum += (grouped[g]/total) * np.log2(grouped[g]/total)

  return -sum

base = getEntropy(data.iloc[:,0])
print('Base:', base)
# Xgt17 = [float(x[1]) < 17 for x in X.values]
# Xgt20 = [float(x[1]) < 20 for x in X.values]
# X = data.iloc[:,1:]
# y = data.iloc[:,0]
print('Entr. 17')
Xlt17 = list(filter(lambda x: float(x[2]) < 17, data.values))
Ylt17 = [x[0] for x in Xlt17]
Xgt17 = list(filter(lambda x: float(x[2]) >= 17, data.values))
Ygt17 = [x[0] for x in Xgt17]
# print(Ylt17)

print('LT')
a = getEntropy(Ylt17)
print(a)
print('GT')
b = getEntropy(Ygt17)
print(b)
print('INFORMATION GAIN', base - a * (len(Xlt17) / len(data)) - b * ( len(Xgt17) / len(data)))
print('________')

print('Entr. 20')
Xone = list(filter(lambda x: float(x[2]) < 20, data.values))
Yone = [x[0] for x in Xone]
Xtwo = list(filter(lambda x: float(x[2]) >= 20, data.values))
Ytwo = [x[0] for x in Xtwo]

print('LT')
a = getEntropy(Yone)
print(a)
print('GT')
b = getEntropy(Ytwo)
print(b)
print('INFORMATION GAIN',base -( len(Xone)/len(data) ) * a - b * ( len(Xtwo) / len(data)))
print('________')

# print('Entr. Brown')
# Xone = list(filter(lambda x: x[1] == 'Brown', data.values))
# Yone = [x[0] for x in Xone]
# Xtwo = list(filter(lambda x: x[1] != 'Brown', data.values))
# Ytwo = [x[0] for x in Xtwo]

# print('Ok')
# a = getEntropy(Yone)
# print(a)
# print('Others')
# b = getEntropy(Ytwo)
# print(b)
# print('INFORMATION GAIN',base -( a + b ) / 2)
# print('________')

# print('Entr. Blue')
# Xone = list(filter(lambda x: x[1] == 'Blue', data.values))
# Yone = [x[0] for x in Xone]
# Xtwo = list(filter(lambda x: x[1] != 'Blue', data.values))
# Ytwo = [x[0] for x in Xtwo]

# print('Ok')
# a = getEntropy(Yone)
# print(a)
# print('Others')
# b = getEntropy(Ytwo)
# print(b)
# print('INFORMATION GAIN',base -( a + b ) / 2)
# print('________')

print('Entr. Green')
Xone = list(filter(lambda x: x[1] == 'Green', data.values))
Yone = [x[0] for x in Xone]
Xtwo = list(filter(lambda x: x[1] != 'Green', data.values))
Ytwo = [x[0] for x in Xtwo]

print('Ok')
a = getEntropy(Yone)
print(a)
print('Others')
b = getEntropy(Ytwo)
print(b)

print('INFORMATION GAIN', base - ((len(Xone)/len(data)) * a + (len(Xtwo)/len(data)) * b))
print('________')
# print('Entr. of GT 20')
# print(getEntropy(Xgt20))
# Xp = list(map(lambda x: x, X.values))



# //////// SOLUTION => x<17


#Nicely done! Did you get an information gain of 0.1126? Here was how I calculated the solution:


print('--------------')
print('ANSWER:')
def two_group_ent(first, tot):                        
    return -(first/tot*np.log2(first/tot) +           
             (tot-first)/tot*np.log2((tot-first)/tot))

tot_ent = two_group_ent(10, 24)                       

print('Base:', tot_ent)

g17_ent = 15/24 * two_group_ent(11,15) + 9/24 * two_group_ent(6,9)                  

answer = tot_ent - g17_ent                            
print('LT 17:', answer)