"""CTC downstream head primitives. Supervised; never part of SSL loss."""
import torch
from torch import nn


class FrozenSequenceCTCHead(nn.Module):
    def __init__(self,input_dim,vocab_size,hidden=128):
        super().__init__()
        self.temporal=nn.Sequential(nn.Conv1d(input_dim,hidden,3,padding=1),nn.GELU(),
                                    nn.Conv1d(hidden,hidden,3,padding=1),nn.GELU())
        self.output=nn.Linear(hidden,vocab_size)

    def forward(self,features):
        if features.requires_grad: raise ValueError("CTC head expects frozen sequence features")
        return self.output(self.temporal(features.transpose(1,2)).transpose(1,2)).transpose(0,1)


def min_ctc_length(target):
    return len(target)+sum(a==b for a,b in zip(target,target[1:]))


def ctc_loss(logits,targets,input_lengths,blank=0):
    if any(blank in target for target in targets): raise ValueError("CTC targets must not contain blank")
    bad=[i for i,(target,length) in enumerate(zip(targets,input_lengths)) if min_ctc_length(target)>int(length)]
    if bad: raise ValueError(f"infeasible CTC alignments: {bad}")
    concatenated=torch.tensor([t for target in targets for t in target],dtype=torch.long,device=logits.device)
    lengths=torch.tensor([len(t) for t in targets],dtype=torch.long)
    return nn.CTCLoss(blank=blank,zero_infinity=False)(logits.float().log_softmax(-1),concatenated,input_lengths,lengths)


def greedy_decode(tokens,blank=0):
    # Collapse repeats before removing blank: A blank A -> A A.
    collapsed=[token for i,token in enumerate(tokens) if i==0 or token!=tokens[i-1]]
    return [t for t in collapsed if t!=blank]


def edit_counts(reference,prediction):
    table=[[(i+j,(0,i,j)) for j in range(len(prediction)+1)] for i in range(len(reference)+1)]
    for i in range(1,len(reference)+1):
        for j in range(1,len(prediction)+1):
            score,counts=table[i-1][j-1]
            mismatch=int(reference[i-1]!=prediction[j-1])
            diag=(score+mismatch,(counts[0]+mismatch,counts[1],counts[2]))
            score,counts=table[i-1][j]; delete=(score+1,(counts[0],counts[1]+1,counts[2]))
            score,counts=table[i][j-1]; insert=(score+1,(counts[0],counts[1],counts[2]+1))
            table[i][j]=min((diag,delete,insert),key=lambda item:item[0])
    s,d,i=table[-1][-1][1]
    return {"substitutions":s,"deletions":d,"insertions":i,"reference_tokens":len(reference),
            "wer":(s+d+i)/len(reference) if reference else None}
