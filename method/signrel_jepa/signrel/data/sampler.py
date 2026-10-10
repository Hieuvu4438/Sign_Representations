"""Resume-stable sampler: augmentations keyed by draw, not worker scheduling."""
import torch
from torch.utils.data import Dataset,Sampler


class SeededDataset(Dataset):
    def __init__(self, dataset, seed):
        self.dataset,self.seed = dataset,seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        index,draw = key
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed((self.seed+draw*1000003) % (2**63-1))
            return self.dataset[index]


class ResumableBatchSampler(Sampler):
    def __init__(self, size,batch_size,seed,start_batch=0,total_batches=1,rank=0,world_size=1):
        self.size,self.batch,self.seed = size,batch_size,seed
        self.start,self.total,self.rank,self.world = start_batch,total_batches,rank,world_size
        self.per_epoch = size//(batch_size*world_size)
        if not self.per_epoch:
            raise ValueError("Dataset smaller than one global batch")

    def __iter__(self):
        last_epoch,order = -1,None
        for step in range(self.start,self.total):
            epoch,position = divmod(step,self.per_epoch)
            if epoch!=last_epoch:
                g = torch.Generator().manual_seed(self.seed+epoch)
                order = torch.randperm(self.size,generator=g).tolist()
                last_epoch = epoch
            begin = (position*self.world+self.rank)*self.batch
            draw_begin = (step*self.world+self.rank)*self.batch
            yield [(order[begin+i],draw_begin+i) for i in range(self.batch)]

    def __len__(self):
        return max(0,self.total-self.start)
