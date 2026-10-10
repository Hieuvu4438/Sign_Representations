from dataclasses import dataclass
import torch
from signrel.models.region_pooler import DEFAULT_EDGES


@dataclass
class MaskSpec:
    visible_tokens: torch.Tensor
    hidden_tokens: torch.Tensor
    metadata: list


class SignMaskGenerator:
    def __init__(self, mixture=None, min_visible_ratio=0.25, hidden_ratio=0.65,
                 span_fraction=0.5, edges=DEFAULT_EDGES, relation_hide_both=False):
        self.mixture = mixture or {"random_multiblock":0.3,"articulator":0.3,"relation_break":0.3,"motion_salient":0.1}
        allowed = {"random_multiblock","articulator","relation_break","motion_salient"}
        if set(self.mixture)-allowed or min(self.mixture.values()) < 0 or sum(self.mixture.values()) <= 0:
            raise ValueError("Invalid masking mixture")
        if not 0 < min_visible_ratio < 1 or not 0 < hidden_ratio < 1 or not 0 < span_fraction <= 1:
            raise ValueError("Invalid mask ratios")
        self.min_visible_ratio,self.hidden_ratio,self.span_fraction = min_visible_ratio,hidden_ratio,span_fraction
        self.edges = tuple(edges)
        self.relation_hide_both = bool(relation_hide_both)

    def sample(self, grid, membership, motion=None, rng=None):
        b,tt,a,hh,ww = membership.shape
        if (tt,hh,ww) != tuple(grid):
            raise AssertionError("Mask membership/grid mismatch")
        names = list(self.mixture)
        probs = torch.tensor(list(self.mixture.values()),dtype=torch.float)
        masks,metadata = [],[]
        # CPU RNG keeps deterministic resume independent of CUDA stream interleaving.
        rand = lambda high: int(torch.randint(high,(),generator=rng))
        for bi in range(b):
            accepted = None
            for attempt in range(64):
                kind = names[int(torch.multinomial(probs,1,generator=rng))]
                hidden = torch.zeros(tt,hh,ww,dtype=torch.bool,device=membership.device)
                info = {"type":kind}
                if kind == "random_multiblock":
                    # Spatial rectangles repeated over time (tube masking), then additional blocks.
                    goal = max(1,int(tt*hh*ww*self.hidden_ratio))
                    for _ in range(128):
                        bh,bw = max(1,rand(hh)+1),max(1,rand(ww)+1)
                        y,x = rand(hh-bh+1),rand(ww-bw+1)
                        hidden[:,y:y+bh,x:x+bw] = True
                        if int(hidden.sum()) >= goal:
                            break
                    # Keep ratio bounded by dropping random excess; no structural target in this branch.
                    # Limit by complete spatial tubes. Dropping flattened
                    # spatiotemporal tokens would destroy tube consistency.
                    ids = hidden[0].flatten().nonzero().flatten()
                    cap = min(max(1,goal//tt),int(hh*ww*(1-self.min_visible_ratio)))
                    if len(ids)>cap:
                        keep = ids[torch.randperm(len(ids),generator=rng,device="cpu")[:cap].to(ids.device)]
                        spatial = torch.zeros(hh*ww,dtype=torch.bool,device=hidden.device)
                        spatial[keep] = True
                        hidden = spatial.reshape(1,hh,ww).expand(tt,-1,-1).clone()
                    info["complete_temporal_tubes"] = True
                else:
                    span = max(1,int(tt*self.span_fraction))
                    if kind == "relation_break":
                        candidates = [(e,s) for e,(u,v) in enumerate(self.edges) for s in range(tt-span+1)
                                      if bool((membership[bi,s:s+span,u].sum((-1,-2))>0).all())
                                      and bool((membership[bi,s:s+span,v].sum((-1,-2))>0).all())]
                        if not candidates:
                            continue
                        edge,start = candidates[rand(len(candidates))]
                        u,v = self.edges[edge]
                        endpoint = (u,v)[rand(2)]
                        other = v if endpoint == u else u
                        info.update(edge=edge,endpoint=endpoint,start=start,end=start+span)
                    else:
                        candidates = [(ar,s) for ar in range(4) for s in range(tt-span+1)
                                      if bool((membership[bi,s:s+span,ar].sum((-1,-2))>0).all())]
                        if kind == "motion_salient":
                            candidates = [(ar,s) for ar,s in candidates if ar < 2]
                        if not candidates:
                            continue
                        if kind == "motion_salient" and motion is not None:
                            weights = torch.tensor([float(motion[bi,s:s+span,ar].sum())+1e-6 for ar,s in candidates])
                            endpoint,start = candidates[int(torch.multinomial(weights,1,generator=rng))]
                        else:
                            endpoint,start = candidates[rand(len(candidates))]
                        info.update(endpoint=endpoint,start=start,end=start+span)
                    hidden[start:start+span] = membership[bi,start:start+span,endpoint]>0
                    if kind == "relation_break" and self.relation_hide_both:
                        hidden[start:start+span] |= membership[bi,start:start+span,other]>0
                        info["endpoints_hidden"] = [endpoint,other]
                    if kind == "relation_break" and not self.relation_hide_both:
                        # Preserve at least one token from the other endpoint at each time.
                        other_visible = (membership[bi,start:start+span,other]>0) & ~hidden[start:start+span]
                        if not bool(other_visible.flatten(1).any(1).all()):
                            continue
                    # Never hide all valid articulators during any target time.
                    remaining = (membership[bi,start:start+span]>0) & ~hidden[start:start+span,None]
                    if not bool(remaining.flatten(1).any(1).all()):
                        continue
                ratio = 1-float(hidden.float().mean())
                if bool(hidden.any()) and ratio >= self.min_visible_ratio:
                    accepted = hidden,info
                    break
            if accepted is None:
                raise AssertionError("No feasible mask: check box overlap, valid regions and minimum visible ratio")
            masks.append(accepted[0].flatten())
            metadata.append(accepted[1])
        hidden = torch.stack(masks)
        return MaskSpec(~hidden,hidden,metadata)
