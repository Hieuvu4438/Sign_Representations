# S1 post-implementation review

`test_model.py`: two tests passed against the actual released checkpoint. All 327
state tensors load strictly. Reproduced `features`, `latent`, `cls_active`, nine
prior outputs and decoder latent outputs are bitwise equal to the original author
forward on a fixed 16-frame input. The adapted constructor cannot silently drop heads.
The final audit also compares training-mode forward at mask ratio .8 and stochastic
depth .1 with restored RNG, covering the actual pretraining encoder path.

64-frame inflation: exact transferred temporal slices [1,3,5], all other slices
zero; a real 64-frame forward succeeds with the same final token grid as 16 frames.
An initial test confused 392 mask units with 1568 final tokens; the oracle was fixed
by comparing against an actual original-input forward and q_pool=2 geometry.

Evidence: `evidence/S1_pytest.txt`. Architecture/checkpoint compatibility PASS.
Training initialization outside the published checkpoint remains an engineering
choice: standard PyTorch head initialization, since the upstream custom-init path
is broken and its historical initializer is unavailable.
