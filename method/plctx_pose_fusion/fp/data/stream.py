import numpy as np
from fp.data.dataset import PhonoWindows


class Windows(PhonoWindows):
    def __getitem__(self,key):
        step,draw,index=key
        return self.item(index,np.random.default_rng([self.seed,step,draw]))


class Batches:
    """Stateless epoch permutation + consumed optimizer-step cursor."""
    def __init__(self,n,batch,seed,start,steps):self.n,self.batch,self.seed,self.start,self.steps=n,batch,seed,start,steps
    def __len__(self):return self.steps-self.start
    def __iter__(self):
        if self.batch>self.n:raise ValueError('Batch exceeds corpus')
        epoch=None
        for step in range(self.start,self.steps):
            current=step//(self.n//self.batch)
            if current!=epoch:
                order=np.random.default_rng([self.seed,current,98765]).permutation(self.n);epoch=current
            begin=step%(self.n//self.batch)*self.batch
            yield [(step,j,int(i)) for j,i in enumerate(order[begin:begin+self.batch])]
