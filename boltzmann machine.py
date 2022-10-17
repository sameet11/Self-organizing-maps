import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

movies=pd.read_csv('ml-1m/movies.dat',sep='::',header=None,engine='python',encoding='latin-1')
users=pd.read_csv('ml-1m/users.dat',sep='::',header=None,engine='python',encoding='latin-1')
rating =pd.read_csv('ml-1m/ratings.dat',sep='::',header=None,engine='python',encoding='latin-1')

training_set=pd.read_csv('ml-100k/u1.base',delimiter='\t')
training_set=np.array(training_set,dtype='int')
test_set=pd.read_csv('ml-100k/u1.test',delimiter='\t')
test_set=np.array(test_set,dtype='int')

nb_users=int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies=int(max(max(training_set[:,1]),max(test_set[:,1])))

def convert(data):
    new_data=[]
    for id_users in range(1,nb_users+1):
        id_movies=data[:,1][data[:,0]==id_users]
        id_ratings=data[:,2][data[:,0]==id_users]
        ratings=np.zeros(nb_movies)
        ratings[id_movies-1]=id_ratings
        new_data.append(ratings)
    return new_data
training_set=convert(training_set)
test_set=convert(test_set)

training_set=torch.FloatTensor(training_set)
test_set=torch.FloatTensor(test_set)
s
training_set[training_set==0]=-1
training_set[training_set==1]=0
training_set[training_set==2]=0
training_set[training_set>=3]=1
test_set[test_set==0]=-1
test_set[test_set==1]==0
test_set[test_set==2]==0

class RBM():
    def __init__(self,nv,nh):
        self.W=torch.randn(nh,nv)
        self.a=torch.randn(1,nh)
        self.b=torch.randn(1,nv)
    def sample_h(self,x):
        wx=torch.mm(x,self.W.t())
        activation=wx+self.a.expand_as(wx)
        p_h_given_v=torch.sigmoid(activation)
        return p_h_given_v , torch.bernoulli(p_h_given_v)
    def sample_v(self,y):
        wy=torch.mm(y,self.W)
        activation=wy+self.b.expand_as(wy)
        p_v_given_h=torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self,vo,vk,pho,phk):
        self.W+=torch.mm((vo.t(),pho)-torch.mm(vk.t(),phk)).t()
        self.b+=torch.sum((vo-vk),0)
        self.a+=torch.sum((pho-phk),0)
nv=len(training_set[0])
nh=100
batch_size=100
rbm=RBM(nv, nh)
nb_epoch=10
for epoch in range(1,nb_epoch+1):
    train_loss=0
    s=0.
    for id_users in range(0,nb_users-batch_size,batch_size):
        vk=training_set[id_users:id_users+batch_size]
        vo=training_set[id_users:id_users+batch_size]
        pho,_=rbm.sample_h(vo)
        for k in range(10):
            _,hk=rbm.sample_h(vk)
            _,vk=rbm.sample_v(hk)
            vk[vo<0]=vo[vo<0]
        phk,_=rbm.sample_h(vk)
        rbm.train(vo, vk, pho, phk)
        train_loss+=torch.mean(torch.abs(vo[vo>=0]-vk[vo>=0]))
        s+=1
    print('epoch: '+str(epoch)+'  loss:'+str(train_loss/s))
test_loss=0
s=0.
for id_users in range(nb_users):
    v=training_set[id_users:id_users+1]
    vt=test_set[id_users:id_users+1]
    if vt[vt>=0]>0:
        _,h=rbm.sample_h(v)
        _,v=rbm.sample_v(h)
        test_loss+=torch.mean(torch.abs(vt[vt>=0]-[vo>=0]))
        s+=1
print('test_loss:  '+str(test_loss/s))
        
    

