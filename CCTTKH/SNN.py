import numpy as np
import pandas as pd
S=2
r=2
m=8
dt = pd.read_csv("/gdrive/My Drive/Colab Notebooks/data/du_lieu_tao.csv")
p=np.array([dt['can nang' ],dt['do chin' ]])
t=np.array([dt['t1'],dt['t2']])
p=p.T
t=t.T
a=np.array([[0,0]])

W = np.array([[1, 0], [2,8]])
b=np.array([[-6,-9]])
k=0
while True:
d=True
k=k+1
print('lần lạp ',k)
for i in range(m):
x=np.array([p[i]])
n=w.dot(x.T)+b.T
for j in range(s):
if(n[j][0]>=0):
a[0][j]=1
else:
a[0][j]=0
if(np.array_equal([t[i]],a) == False):
e=t[i]-a
e1=e.T
W=W+e1.dot(x)
b=b+e
d=False
print('w=',w)
print('b=',b)
if(d == True):
break

f=np.array([[9,2]])
n=w.dot(f.T)+b.T#w.x+b
for j in range(s):
if(n[j][0]>=0):
a[0][j]=1
else:
a[0][j]=0
print('lop cua f la:',a)