"""DuoRec same-target semantic views from training examples only."""
import torch

class IndexDataset(torch.utils.data.Dataset):
    def __init__(self,base):self.base=base
    def __len__(self):return len(self.base)
    def __getitem__(self,index):return index

class SemanticCollator:
    def __init__(self,base):
        self.base=base
        targets=base.positive_items
        self.order=torch.argsort(targets,stable=True)
        sorted_targets=targets[self.order]
        self.start=torch.searchsorted(sorted_targets,targets,right=False)
        self.count=torch.searchsorted(sorted_targets,targets,right=True)-self.start
        self.rank=torch.empty_like(self.order);self.rank[self.order]=torch.arange(len(targets))
    def sample_indices(self,ids):
        counts=self.count[ids];own=self.rank[ids]-self.start[ids]
        offset=(torch.rand(len(ids))*(counts-1).clamp(min=1)).long()
        offset=offset+(offset>=own).long()
        offset=torch.where(counts>1,offset,own)
        return self.order[self.start[ids]+offset]
    def __call__(self,indices):
        ids=torch.tensor(indices,dtype=torch.long);other=self.sample_indices(ids);b=self.base
        assert torch.equal(b.positive_items[ids],b.positive_items[other])
        return b.sequences[ids],b.sequence_lengths[ids],b.positive_items[ids],b.sequences[other],b.sequence_lengths[other]

class CompactHistoryDataset(torch.utils.data.Dataset):
    """Same sliding windows as the eager gSASRec loader, zero-based IDs."""
    def __init__(self,histories,width,padding):
        import numpy as np
        self.histories=[torch.tensor(v,dtype=torch.long) for v in histories.values() if len(v)>=2]
        self.width=width;self.padding=padding
        self.counts=np.array([max(1,len(v)-width+1) for v in self.histories],dtype=np.int64)
        self.ends=np.cumsum(self.counts)
    def __len__(self):return int(self.ends[-1]) if len(self.ends) else 0
    def __getitem__(self,index):
        import numpy as np
        group=int(np.searchsorted(self.ends,index,side='right'));begin=0 if group==0 else int(self.ends[group-1])
        offset=index-begin;values=self.histories[group][offset:offset+self.width]
        result=torch.full((self.width,),self.padding,dtype=torch.long);result[:len(values)]=values
        return result

class HistoryCollator:
    def __init__(self,n_items,negatives):self.n_items=n_items;self.negatives=negatives
    def __call__(self,samples):
        positives=torch.stack(samples);b,l=positives.shape
        valid=positives[:,1:]!=self.n_items
        negatives=torch.randint(self.n_items,(b,l-1,self.negatives))
        prefix=torch.arange(l)[None,:]<=torch.arange(1,l)[:,None]
        while True:
            collisions=((negatives[:,:,:,None]==positives[:,None,None,:])&prefix[None,:,None,:]).any(-1)&valid[:,:,None]
            if not collisions.any():break
            negatives[collisions]=torch.randint(self.n_items,(int(collisions.sum()),))
        negatives.masked_fill_(~valid[:,:,None],self.n_items)
        return positives,negatives
