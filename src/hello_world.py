a=1
print ('hello,world\n',a)
for i in range(0,5):
    print(i)
#s=float(input('hello'))
#print(s)
classmates=['Marry','Jack','LiLei']
student_name=classmates.pop(1)
print(student_name,len(classmates))
L = [
    ['Apple', 'Google', 'Microsoft'],
    ['Java', 'Python', 'Ruby', 'PHP'],
    ['Adam', 'Bart', 'Lisa']
]
print(L[1][1])
temp=int(input('please input a number:'))
if temp>5:
    print('temp>5')
elif temp<4:
    print('nihaoma')
    print('wohenhao')
else:
    print('else')
d = {'Michael': 95, 'Bob': 75, 'Tracy': 85}
#d['LiJingYu']=59
print(d.get('LiJingYu',60))
#print(d['LiJingYu'])

def enroll(name, gender, age=6, city='Beijing'):
    print('name:', name)
    print('gender:', gender+1)
    print('age:', age)
    print('city:', city)
enroll('Bob', 3, 7)
enroll('Adam', 2, city='Tianjin')