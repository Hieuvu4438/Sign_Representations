# REJECTANALTEST001 — bootstrap/compact-curve manual software gate

CPU60s/GPU0/HPO0/seed1701/full2MiB. No numeric signfeatures/quality. P007_v1
unchanged; bind D081/hypothesis/protocol and exactimplementation before run.
Five manualgroups: cluster weightedqueries, equal-seed mean, undefineddraws,
fixedcluster multiplicities, finitecompactcurve encoding/schema.

Groups[A,A,B], acceptance[1,0,0], allmasktrue; multiplicities[2,0],[1,1],[0,2]
give rates1/2,1/3,0 (not equalclustermean1/4 onbalanced draw). Add secondseed
acceptance[1,0,1], mask[1,0,1]; meanseed rates3/4,2/3,1/2. Restrict firstmask
to[A,A] then draw[0,2] undefined, preserve invalidrep and return no95%CI instead
of dropping/resampling/meanzero. Fixed20multinomialdraws sum2 each.
Curve +inf/0counts then .8knownaccept2/correct1/unknown0, .2knownaccept2/
correct1/unknown2 ->finite rows[0,0,0,0],[.8,2,1,0],[.2,2,1,2], firstkind
reject_all explicitly in scientificmetadata. Empty/nonzerorejectpoint/invalid
multiplicities raiseValueError. These are representation/statisticssoftware
checks, not uncertainty guarantees or scientificexperiment/trial results.
