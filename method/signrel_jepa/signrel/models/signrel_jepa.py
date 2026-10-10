import copy
import torch
import torch.nn.functional as F
from torch import nn
from .backbone_adapter import make_backbone, positions
from .region_pooler import RegionPooler, token_membership, DEFAULT_EDGES, ARTICULATORS
from .relation_projector import RelationProjector
from .relation_predictor import QueryPredictor
from .temporal_pyramid import temporal_targets
from .ema import freeze, update_ema
from signrel.masks.base import SignMaskGenerator
from signrel.losses.latent import latent_loss
from signrel.losses.regularization import variance_covariance
from signrel.data.manifest import assert_no_labels


class SignRelJEPA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        mc = cfg["model"]
        self.loss_cfg = cfg.get("loss",{})
        self.student_encoder = make_backbone(mc["backbone"])
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        dim = self.student_encoder.dim
        self.dim = dim
        self.edges = tuple(tuple(ARTICULATORS.index(v) if isinstance(v,str) else v for v in edge)
                           for edge in mc.get("relation_edges",DEFAULT_EDGES))
        if not self.edges or any(u==v or min(u,v)<0 or max(u,v)>3 for u,v in self.edges):
            raise ValueError("Invalid relation graph")
        self.scales = mc.get("temporal_scales",[{"name":"short","kernel":1,"weight":1.}])
        rd = mc.get("relation_dim",256)
        self.student_relation_projector = RelationProjector(dim,rd,mc.get("projector","concat_diff_product"))
        self.teacher_relation_projector = copy.deepcopy(self.student_relation_projector)
        pc = mc.get("predictor",{})
        # Decode two visual latent endpoints, then use the *student* relation projector.
        # This trains R_student via relation prediction and gives its EMA a trained source.
        self.relation_predictor = QueryPredictor(dim,2*dim,len(self.edges),**pc)
        self.articulator_predictor = QueryPredictor(dim,dim,4,**pc)
        self.generic_predictor = QueryPredictor(dim,dim,1,**pc)
        self.region_pooler = RegionPooler()
        self.mask_generator = SignMaskGenerator(edges=self.edges,**cfg.get("masking",{}))
        freeze(self.teacher_encoder)
        freeze(self.teacher_relation_projector)

    def train(self, mode=True):
        super().train(mode)
        self.teacher_encoder.eval()
        self.teacher_relation_projector.eval()
        return self

    @torch.no_grad()
    def ema_update(self, momentum):
        update_ema(self.teacher_encoder,self.student_encoder,momentum)
        update_ema(self.teacher_relation_projector,self.student_relation_projector,momentum)

    @torch.no_grad()
    def build_targets(self, video, membership):
        teacher = self.teacher_encoder.forward_tokens(video)
        grid = teacher["grid_shape"]
        dense = teacher["tokens"].reshape(video.shape[0],*grid,self.dim)
        nodes,node_valid = self.region_pooler(dense,membership)
        u,v = zip(*self.edges)
        rel = self.teacher_relation_projector(nodes[:,:,u],nodes[:,:,v])
        valid = node_valid[:,:,u] & node_valid[:,:,v]
        return teacher,nodes,node_valid,rel,valid

    def forward(self, batch):
        assert_no_labels(batch)
        video,meta = batch["video"],batch["articulator_meta"]
        grid = self.student_encoder.grid_shape(video)
        b,tt,hh,ww = video.shape[0],*grid
        membership = token_membership(meta["boxes"],meta["confidence"],grid,
                         self.cfg.get("articulators",{}).get("confidence_threshold",0.5),
                         self.cfg.get("articulators",{}).get("membership","center"))
        needs_regions = self.loss_cfg.get("relation_weight",0)>0 or self.loss_cfg.get("articulator_weight",0)>0
        if needs_regions and not bool(membership.sum()>0):
            raise AssertionError("Zero-valid-articulator batch")
        motion = meta.get("motion")
        if motion is not None:
            motion = motion.reshape(b,tt,-1,4).mean(2)
        mask = self.mask_generator.sample(grid,membership,motion)
        teacher,nodes,node_valid,relations,rel_valid = self.build_targets(video,membership)
        student = self.student_encoder.forward_tokens(video,mask.visible_tokens)
        ctx,padding = student["tokens"],student["padding_mask"]
        loss = ctx.sum()*0
        metrics = {}
        outputs = {"mask":mask,"membership":membership,"node_valid":node_valid}
        kind = self.loss_cfg.get("type","l1")
        normalize = self.loss_cfg.get("normalize","layernorm")
        hidden_grid = mask.hidden_tokens.reshape(b,tt,hh,ww)
        affected = ((membership>0) & hidden_grid[:,:,None]).flatten(-2).any(-1)
        rw = self.loss_cfg.get("relation_weight",0.)
        if rw:
            per_scale = []
            for si,scale in enumerate(self.scales):
                kernel = scale["kernel"]
                target,valid = temporal_targets(relations,rel_valid,kernel)
                # Predict only relations whose target window intersects corrupted endpoints.
                u,v = zip(*self.edges)
                changed = (affected[:,:,u] | affected[:,:,v]).unfold(1,kernel,1).any(-1)
                valid = valid & changed
                nq = target.shape[1]
                edge_ids = torch.arange(len(self.edges),device=video.device).repeat(nq)
                times = torch.arange(nq,device=video.device).repeat_interleave(len(self.edges))
                coords = torch.stack((times/max(tt-1,1),torch.zeros_like(times),torch.zeros_like(times),
                                     torch.full_like(times,kernel)/tt),-1).float()
                endpoints = self.relation_predictor(ctx,padding,edge_ids[None].expand(b,-1),coords[None].expand(b,-1,-1))
                a,z = endpoints.chunk(2,-1)
                pred = self.student_relation_projector(a,z).reshape_as(target)
                lr = latent_loss(pred,target,valid,kind,normalize,distributed=True)
                loss = loss + rw*scale.get("weight",1.)*lr
                per_scale.append(lr.detach())
                metrics[f"loss_relation_{scale['name']}"] = float(lr.detach())
                for e,(u,v) in enumerate(self.edges):
                    ve = valid[:,:,e]
                    metrics[f"edge_{u}_{v}_{scale['name']}_loss"] = 0.
                    metrics[f"edge_{u}_{v}_{scale['name']}_target_std"] = 0.
                    metrics[f"edge_{u}_{v}_{scale['name']}_valid_queries"] = float(ve.sum())
                    if bool(ve.any()):
                        metrics[f"edge_{u}_{v}_{scale['name']}_loss"] = float(latent_loss(pred[:,:,e],target[:,:,e],ve,kind,normalize).detach())
                        metrics[f"edge_{u}_{v}_{scale['name']}_target_std"] = float(target[:,:,e][ve].float().std(0,unbiased=False).mean())
                metrics[f"valid_relation_queries_{scale['name']}"] = float(valid.sum())/b
                outputs[f"pred_relation_{si}"] = pred
                outputs[f"target_relation_{si}"] = target
                outputs[f"query_valid_{si}"] = valid
            metrics["loss_relation"] = float(torch.stack(per_scale).mean())
        aw = self.loss_cfg.get("articulator_weight",0.)
        if aw:
            ids = torch.arange(4,device=video.device).repeat(tt)
            times = torch.arange(tt,device=video.device).repeat_interleave(4)
            coords = torch.stack((times/max(tt-1,1),times*0,times*0,torch.ones_like(times)/tt),-1).float()
            pred = self.articulator_predictor(ctx,padding,ids[None].expand(b,-1),coords[None].expand(b,-1,-1)).reshape_as(nodes)
            valid = node_valid & affected
            la = latent_loss(pred,nodes,valid,kind,normalize,distributed=True)
            loss = loss + aw*la
            metrics["loss_articulator"] = float(la.detach())
        gw = self.loss_cfg.get("generic_weight",0.)
        if gw:
            # Predict masked visual patch latents with location queries; cap only for memory.
            cap = self.loss_cfg.get("max_patch_queries",256)
            indices = []
            for hidden in mask.hidden_tokens:
                ids = hidden.nonzero().flatten()
                selection = torch.randperm(len(ids),device="cpu")[:cap].to(ids.device)
                indices.append(ids[selection])
            max_q = max(len(i) for i in indices)
            ids = torch.zeros(b,max_q,dtype=torch.long,device=video.device)
            coords = video.new_zeros(b,max_q,4)
            target = ctx.new_zeros(b,max_q,self.dim)
            valid = torch.zeros(b,max_q,dtype=torch.bool,device=video.device)
            pos = positions(grid,video.device,video.dtype)
            for bi,i in enumerate(indices):
                coords[bi,:len(i),:3] = pos[i]
                target[bi,:len(i)] = teacher["tokens"][bi,i]
                valid[bi,:len(i)] = True
            pred = self.generic_predictor(ctx,padding,ids,coords)
            lg = latent_loss(pred,target,valid,kind,normalize,distributed=True)
            loss = loss + gw*lg
            metrics["loss_generic"] = float(lg.detach())
        rc = self.loss_cfg.get("regularization",{})
        if rc.get("enabled",False):
            z = student["global_feature"]
            if torch.distributed.is_initialized():
                from torch.distributed.nn.functional import all_gather
                z = torch.cat(all_gather(z))
            lv,lc = variance_covariance(z,rc.get("gamma",1.))
            reg = rc.get("variance_weight",1.)*lv+rc.get("covariance_weight",0.004)*lc
            loss = loss + reg
            metrics["loss_reg"] = float(reg.detach())
        if not (rw or aw or gw):
            raise AssertionError("At least one latent prediction objective is required")
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite SSL loss")
        target_values = relations[rel_valid] if rw else nodes[node_valid] if aw else teacher["tokens"].flatten(0,1)
        metrics.update(loss_total=float(loss.detach()),masked_token_ratio=float(mask.hidden_tokens.float().mean()),
                       student_embedding_std=float(student["global_feature"].detach().float().std(0,unbiased=False).mean()),
                       teacher_embedding_std=float(teacher["global_feature"].float().std(0,unbiased=False).mean()),
                       teacher_target_std=float(target_values.float().std(0,unbiased=False).mean()) if len(target_values) else 0.)
        if b>1:
            with torch.autocast(video.device.type,enabled=False):
                normalized = F.normalize(student["global_feature"].detach().float(),dim=-1)
                cosine = normalized@normalized.T
            metrics["between_sample_cosine"] = float(cosine[~torch.eye(b,dtype=torch.bool,device=video.device)].mean())
        for name in self.mask_generator.mixture:
            metrics[f"mask_frequency_{name}"] = sum(m["type"]==name for m in mask.metadata)/b
        outputs.update(loss=loss,metrics=metrics,global_feature=student["global_feature"])
        return outputs

    @torch.no_grad()
    def encode_video(self, video, teacher=False):
        """RGB-only feature extraction; no localization or relation metadata required."""
        encoder = self.teacher_encoder if teacher else self.student_encoder
        z = encoder.forward_tokens(video)
        b = video.shape[0]
        tt,hh,ww = z["grid_shape"]
        grid = z["tokens"].reshape(b,tt,hh,ww,self.dim)
        feature = F.layer_norm(grid.mean((1,2,3)),(self.dim,))
        return {"token_features":grid,"frame_or_tube_features":grid.mean((2,3)),
                "global_embedding":F.normalize(feature,dim=-1)}
