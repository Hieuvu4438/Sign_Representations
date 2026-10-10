# Decisions

- New isolated folder `method/signrel_temporal`, independent package `signrel`, distribution `signrel-temporal`; execute only inside its environment to avoid proposal6 package-name collision.
- Preserve existing methods/data/checkpoints and unrelated running GPU jobs.
- Use working Python 3.13.5 environment packages in new isolated core/perception venvs; no changes to the existing interpreter environment. Both actual runtime dependency locks and forward tests are required.
- Initial global budget is bounded at 2 GPU hours across all pilot runs, not 2 hours multiplied by arbitrary trials. Full scale remains disabled.
- How2Sign local recording subset is preferred for independent SSL; ASL Citizen official held-out splits supply evaluation labels separately. Final subset/protocol will be frozen before feature-based decisions. No strict unseen-gloss/SSL signer-disjoint claim with unknown identities.

## Pilot perception gate

Twenty real clips reviewed; hand valid coverage ~.826/.879, body~.995, nonconstant relation targets std~.135–.150. Remaining crop/landmark errors, including a large forearm crop under crossing, are disclosed. Freeze current perception for controls; no interpolation or silent teacher replacement. Full47 train caches audit clean numerically. Segmented How2Sign clips contribute a segmentation prior; lexical identities and continuous signer IDs remain unknown.

## Measured pilot plan before real scores

Initial CUDA profile: .663 seconds/update for full state+relation forward/backward/sampling/AdamW, effective batch32, BF16, peak measured allocated VRAM~193 MB (teacher extraction and other users' GPU memory are separate). Paired seed17 A1/A4 planned1000 updates each, expected22.1 minutes plus overhead. Main selection remains last fixed-budget checkpoint, no dev score tuning. Cache47 SSL train +3 SSL dev and120 gallery/dev videos; locked test scores remain unopened. A phase7 matrix gets a separate measured allocation after this pilot and stays within the existing global cap.

## Frozen dev reference and evaluation coverage

A0 raw teacher+body K=1 dev retrieval: macro/micro R@1 .1375, macro R@5 .35, MRR .25737, query coverage1.0 and40 gallery glosses. This is not the previous-method protocol and does not establish relation utility. ASL Citizen canonical hand-valid fractions .219/.259 vs How2Sign ~.83/.88 are observed-frame coverage, not annotated detector recall; idle/out-of-frame hands contribute. No preprocessing change after controls start. Five additional evaluation previews reviewed; crop/missingness remains a material limitation.
