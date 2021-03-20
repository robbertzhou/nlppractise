#感知机的实现
import numpy as np

#与门
def AND(x1,x2):
    w1,w2,theta = 0.5,0.5,0.7
    tmp = w1*x1 + w2*x2
    if tmp <= theta:
        return 0
    else:
        return 1

#与非门
def NAND(x1,x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = w1 * x1 + w2 * x2
    if tmp > theta:
        return 0
    else:
        return 1

def OR(x1,x2):
    if x1 != x2:
        return 1
    else:
        return 0
print("与门结果：")
print(AND(0,0))
print(AND(0,1))
print(AND(1,0))
print(AND(1,1))

print("与非门结果：")
print(NAND(0,0))
print(NAND(0,1))
print(NAND(1,0))
print(NAND(1,1))

print("或门结果：")
print(OR(0,0))
print(OR(0,1))
print(OR(1,0))
print(OR(1,1))

