# coding=utf-8
'''
Created on 2018年1月31日

@author: lyq
'''

import re

t = '21:5:0'
m = re.match(r'^(0|1|2[0-9])\:(0|1|2|3|4|5)\:(0|1|2|3|4|5)$', t)
print(m.groups())
print(m)