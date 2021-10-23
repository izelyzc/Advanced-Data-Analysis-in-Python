# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 15:07:59 2021

@author: izely
"""

#write a recursive function to determine the nth number of the Fibonacci sequence
#put the function in a class

class Fib(object):
    def __init__(self,n):
        self.n=n
        self.FibN=self.Fibonacci(n)
        self.FibDavid=self.Fibonacci_David(n)
    def Fibonacci(self,n):
    	if n<0:
    		print("Incorrect input")
    	# First Fibonacci number is 0
    	elif n==0:
    		return 0
    	# Second Fibonacci number is 1
    	elif n==1:
    		return 1
    	else:
    		return self.Fibonacci(n-1)+self.Fibonacci(n-2)
     def Fibonacci_David(self,n):
         if n==1 or n==2: return 1
         return Fib(n-1) + Fib(n-2)         
        

    
X=Fib(9)


    