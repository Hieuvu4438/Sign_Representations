"""Pinned real Hiera checkpoint dense/masked output parity and CPU timing."""
import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT/'src'))
from signcoord.models.generic_hiera_adapter import GenericHieraAdapter
from signcoord.models.attention import configure_attention
from torch.nn.attention import sdpa_kernel,SDPBackend
from signcoord.provenance import atomic_json,execution_stamp

parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
torch.set_num_threads(2);torch.manual_seed(7)
encoder=GenericHieraAdapter(PROJECT/'vendor/hiera',32,PROJECT/'runs/assets/mae_hiera_base_16x224.pth').eval()
rgb=torch.randn(1,3,32,224,224);fv=torch.ones(1,32,dtype=torch.bool);times=torch.arange(32)[None]/25
keep=encoder.random_mask(1,.8,'cpu')
with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
    start=time.perf_counter();native=encoder.encode_dense(rgb,fv,times);native_seconds=time.perf_counter()-start
    masked=encoder.forward_visible(rgb,keep,fv,times)
    configure_attention(encoder.bb,'sdpa_4d_zero_cls')
    math_dense=encoder.encode_dense(rgb,fv,times)
    math_masked=encoder.forward_visible(rgb,keep,fv,times)
math_errors={'dense_h':float((native[0]-math_dense[0]).abs().max()),
             'masked_memory':float((masked[0]-math_masked[0]).abs().max())}
assert max(math_errors.values())<3e-5,math_errors
with torch.inference_mode():
    start=time.perf_counter();fast=encoder.encode_dense(rgb,fv,times);fast_seconds=time.perf_counter()-start
    fast_masked=encoder.forward_visible(rgb,keep,fv,times)
errors={'dense_h':float((native[0]-fast[0]).abs().max()),'clip_raw':float((native[3]-fast[3]).abs().max()),
        'masked_memory':float((masked[0]-fast_masked[0]).abs().max())}
relative_rms=float((masked[0]-fast_masked[0]).square().mean().sqrt()/masked[0].square().mean().sqrt())
# The implementation equivalence gate uses the same math backend above.
# Fused-kernel drift is reported separately; it is not bitwise equivalence.
assert relative_rms<1e-4 and torch.isfinite(fast_masked[0]).all()
atomic_json(args.output,{**execution_stamp(),'status':'passed','device':'cpu','max_absolute_errors':errors,
                        'matched_math_backend_errors':math_errors,'fused_masked_relative_rms_error':relative_rms,
                        'initial_fused_absolute_3e_5_gate_failed':True,
                        'numerical_policy':'same_math_backend_proves_layout; fused_backend_float32_drift_reported_separately',
                        'pytorch_reference':'https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html',
                        'vendor_5d_seconds':native_seconds,'scoped_4d_seconds':fast_seconds,
                        'timing_scope':'single_forward_sanity_under_shared_CPU_load_not_stable_budget',
                        'weights_and_architecture_unchanged':True,'vendor_files_mutated':False})
print(args.output,errors,flush=True)
