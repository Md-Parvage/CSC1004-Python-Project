import math as m
def c(x, y): 
    return m.sqrt(((1.5-x)**2)+((3.5-y)**2))

a = [(2,10), (2,5), (8,4), (5,8), (7,5), (6,4), (1,2), (4,9)]

for x, y in a:
    print(c(x, y))
