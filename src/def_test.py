# coding=gbk
'''
Created on 2018年1月27日

@author: lyq
'''

def count():
    fs=[]
    for i in range(1,4):
        def f():
            return i*i
        fs.append(f())
    return fs

f1,f2,f3=count()

class student(object):
    def __init__(self,name,score):
        self.__name=name
        self.score=score
        
    def get_name(self):
        return self.__name
s=student('liumei',92)

d=['a','b','c']
for i in d:
    print(i)
