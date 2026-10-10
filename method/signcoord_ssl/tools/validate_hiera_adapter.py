"""Real checkpoint CPU compatibility and native index transform verification."""
import argparse
import gc
import sys
import time
from pathlib import Path

import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/"src"))
from signcoord.models.hiera_adapter import HieraAdapter, visible_indices
from signcoord.models.generic_hiera_adapter import GenericHieraAdapter
from signcoord.provenance import atomic_json,execution_stamp,sha256_file


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",required=True)
    parser.add_argument("--backend",choices=("signrep","generic"),default="signrep"); args=parser.parse_args()
    repo=PROJECT.parents[1]; reports=[]
    torch.set_num_threads(2)
    for frames in (16,32,64):
        start=time.perf_counter(); torch.manual_seed(7)
        checkpoint=(repo/"checkpoints/signrep/ckpt.pt" if args.backend=="signrep" else PROJECT/"runs/assets/mae_hiera_base_16x224.pth")
        enc=(HieraAdapter(repo/"third_party/SignRep",frames,checkpoint) if args.backend=="signrep" else
             GenericHieraAdapter(PROJECT/"vendor/hiera",frames,checkpoint)).eval()
        mask=enc.random_mask(2,.80,"cpu")
        # Match native indexing exactly: Unroll then masked selection then
        # simulated stride pooling then final Reroll. Labels encode MU origins
        # and local child positions rather than max-pooling arbitrary IDs.
        t,h,w=enc.bb.tokens_spatial_shape
        tt,yy,xx=torch.meshgrid(torch.arange(t),torch.arange(h),torch.arange(w),indexing="ij")
        # q_pool2 -> 4x4 source token blocks per final token. Coordinates remain
        # constant within each pooled block so pooling has exact known metadata.
        coords=torch.stack((tt,yy//4,xx//4),-1).reshape(1,-1,3).float().expand(2,-1,-1)
        native=enc.bb.unroll(coords)
        native=native[mask[...,None].tile(1,enc.bb.mu_size,3)].reshape(2,-1,3)
        for _ in range(enc.bb.q_pool): native=native.reshape(2,4,-1,3).amax(1)
        rerolled=enc.bb.reroll(native,enc.bb.stage_ends[-1],mask=mask)
        _,expected,_=visible_indices(mask,enc.mu_grid,tuple(rerolled.shape[2:5]))
        error=(rerolled.reshape(2,-1,3)-expected).abs().max().item()
        assert error == 0, (frames,error)
        # T64 is a rebuild/load/index check; full model compute is for primary T16/T32.
        record={"frames":frames,"load_report":enc.load_report,"native_index_error":error,
                "dense_forward_status":"not_run" if frames==64 else "passed"}
        if frames != 64:
            rgb=torch.randn(1,3,frames,224,224); fv=torch.ones(1,frames,dtype=torch.bool); times=torch.arange(frames)[None]/25
            with torch.inference_mode():
                h,ht,hv,z=enc.encode_dense(rgb,fv,times)
                m=enc.random_mask(1,.8,"cpu")
                memory,coord,valid,pooled=enc.forward_visible(rgb,m,fv,times)
                # Full visible mask must equal dense grid sorted by mask-unit,
                # not by flattened global raster order.
                full=torch.ones_like(m)
                all_memory,_,_,_=enc.forward_visible(rgb,full,fv,times)
                flat,_,_=visible_indices(full,enc.mu_grid,(1,2,2))
                grid=enc._forward_grid(rgb,None).reshape(1,-1,enc.dim)
                full_error=(all_memory-grid.gather(1,flat[...,None].expand(-1,-1,enc.dim))).abs().max().item()
            assert h.shape == (1,frames//2,768)
            assert torch.equal(ht,times[:,::2]) and torch.isfinite(z).all()
            assert full_error < 1e-5
            record.update(sequence_shape=list(h.shape),memory_shape=list(memory.shape),
                          full_keep_vs_dense_max_error=full_error,mask_true_means_keep=True)
        record["elapsed_seconds"]=time.perf_counter()-start; reports.append(record)
        del enc; gc.collect()
    atomic_json(args.output,{**execution_stamp(),"status":"passed","device":"cpu","torch":torch.__version__,
                  "backend":args.backend,"checkpoint_sha256":sha256_file(checkpoint),"checks":reports})
    print(args.output)


if __name__ == '__main__': main()
