# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 20:31:05 2021

@author: izely
"""
class Clock(object):
    def __init__(self, hour, minutes):
        self.minutes = minutes
        self.hour = hour

    @classmethod
    def at(cls, hour, minutes=0):
        return cls(hour, minutes)

    def __str__(self):
        return self.hour+":"+self.minutes
    
    def __add__(self,minutes):
        time=(int(self.hour)*60 + int(self.minutes) + int(minutes))%60 
    
    def __sub__(self,minutes):
    
    def __eq__(self, other):
       # if two clock are equal --- true
       return (self.hour==other.hour and self.minutes==other.minutes)
    
    def __ne__(self, other):
          # if two clock are  not equal --- true