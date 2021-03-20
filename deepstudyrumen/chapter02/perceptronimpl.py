import numpy as np

def AND(x1,x2):
    w1,w2,theta = 0.5,0.5,0.7
    tmp = x1 * w1 + x2 * w2
    print("middle :" , (tmp - theta))
    if tmp <= theta:
        return 0
    else:
        return 1

def npand(x1,x2):
    w = np.array([0.5,0.5])
    b = -0.7
    arr = np.array([x1,x2])
    tmp = np.sum(w*arr)
    tmp = tmp + b
    print("middle :" , tmp)
    if tmp <=0:
        return 0
    else:
        return 1


def notand(x1,x2):
    w = np.array([0.5,0.5])
    b = 0.7
    tmp = np.sum(w * np.array([x1,x2])) + b
    if tmp <=0 :
        return 0
    else:
        return 1



print(AND(1,1))
print(npand(1,1))