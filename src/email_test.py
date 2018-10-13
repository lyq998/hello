#coding: utf-8
from email.mime.text import MIMEText
import smtplib
msg = MIMEText('hello, send by Python...', 'plain', 'utf-8')

from_addr = 'm18580590062@163.com'
password ='lll19980129000'

to_addr = '1059708311@qq.com'

try:
    server = smtplib.SMTP("smtp.163.com", 25) 
    server.set_debuglevel(1)
    server.login(from_addr, password)
    server.sendmail(from_addr, [to_addr], msg.as_string())
    print('done')
except smtplib.SMTPException as e:
    print('Error: Case:%s'%e)
