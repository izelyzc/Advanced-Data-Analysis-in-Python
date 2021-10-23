# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 14:34:53 2021

@author: izely
"""

class Calendar(object):
    def __init__(self, year, month, day):
        self.day ='0'*(2-len(str(day)))+str(day)
        self.month ='0'*(2-len(str(month)))+str(month)
        self.year = str(year)
        
    def __str__(self):
        return self.year+"."+self.month+"."+self.day        
    
    def __repr__(self):
        return self.__str__()
   # @classmethod
    def __add__(self,day):
        date=(int(self.year)*360+int(self.month)*30+int(self.day)+day)%(360*30)
        return Calendar(date//360,date//30,date%30)
        
    def __sub__(self,day):
        return self+((-1)*day) ## self.__add__((-1)*day)
            
    def __eq__(self, other):
        return (self.day==other.day and self.month==other.month and self.year==other.year )  
    
    def __ne__(self, other):
        return not self==other
    
    # def tmp(self):
    #     print('Deneme', self.day)

    # def tmp2():
    #     print('Deneme2')

Calendar.tmp2()
        
x=Calendar(2011,2,12)   

x.__add__(5)
x.__sub__(5)
x.__repr__()

x.tmp()
Calendar.tmp(x) #x.tmp()