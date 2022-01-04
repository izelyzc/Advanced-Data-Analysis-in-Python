# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 14:42:28 2021

@author: izely
"""
myletters=['orange',2,4,'lemon']
myletters.append(5)
myletters[-1]
type(myletters[-1])
myletters[0]='apple'
myletters[len(myletters)]

tup=(1,6,5,'apple')


## Fibonacci Exercise
list_fibo=[1,1]
for i in range(2,10):
    x=list_fibo[i-1]+list_fibo[i-2]
    list_fibo.append(x)
    
    
fib=[1,1]
while len(fib)<10:
    fib.append(fib[-1]+fib[-2])
    
def Fibonacci_Series(n):
    list_fibo=[1,1]
    for i in range(2,n):
        x=list_fibo[i-1]+list_fibo[i-2]
        list_fibo.append(x)
    return(list_fibo)  
 
    
fib100=Fibonacci_Series(100)    

""" 2^3   2^2  2^1  2^0
    1     0    0     0   = 8
          1    0     1   = 5
          1    1     1   = 7
          """
          
def binarify(num):
    """ convert positive int to base 2 """
    if num<=0: return '0'
    digits =[]
    return ''.join(digits)


def base_to_int(string,num):
    
    
          


 
