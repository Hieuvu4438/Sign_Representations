# OSCRREFTEST001 — pinned reference/manual oracles v1

Diagnostic/software only, before any H007 rejection-quality access.
Seed1701, CPU60seconds/GPU0, HPO0, complete runtime cap2MiB.
Exclusive64KiB sourcecapture, expected utils33525B and LICENSE1518B, SHA bound
by D076 and config. Snapshot complete source/license and wrapper dependencies.
Extract only audited compute_oscr AST from pinned utils, never import module.
Function signature(targets,logits,os_score), no decorators/imports; NumPy plus
sorted/range/len. No model/data/public training code run or environment install.

Three fixture vectors, all2known/2unknown, score higher means acceptance.
logits eachrow[1,0]; np.argmax predictsclass0; unknown target−1.

| ID | targets | scores | manual joint area | predicted pinned result |
| --- | --- | --- | ---: | ---: |
| all_known_wrong | [1,1,−1,−1] | [.8,.8,.2,.2] | 0 | .25 |
| perfect_separated | [0,0,−1,−1] | [.8,.8,.2,.2] | 1 | 1 |
| wrong_tied_scores | [1,1,−1,−1] | [.9,.9,.9,.9] | 0 | .25 |

Oracle: no known correct gives CCR(theta)=0 at everythreshold, any area under
CCR-versus-FPR therefore0 irrespective of score ties. Perfectseparation with
correct predictions reachesCCR1 before firstunknown accepted and has area1.
Pinned calculation inspected: FP/(FP+Nunknown) max2/(2+2)=.5; forced endpoint
(1,1) afterCCR0 contributes(1−.5)*(0+1)/2=.25. This FPR diagnostic is source-
expression/manual arithmetic, not a returned upstream curve (function returns
only scalar). No assertion that paper scalarAUC convention was itself defined
in2018: paper defines curve/operating points, this area oracle is unambiguous
for these fixtures. Original Eq3 known>theta vs unknown>=theta doesn't change
wrong-known oracle or separated area. No general parity across tie policies.

Execute exact function unchanged, report expected/observed/manual values and
absolute float tolerance1e−12. Expected disagreements2 imply reference rejected
for H007 evaluator; fixtureauditpasses if observed sourcebehavior matches fixed
prediction. Unexpected reference result fails audit, preserving trace/artifacts.
Outputs reference_tests.json and ordinary bounded context/config/source/status/
logs/manifest. Actual signed interpretation in later decision; never edit bound
protocol/source/decision/config after registration or completed runtime outputs.

Scope excludes released prediction/paper score reproduction, all public code,
sign-language task performance or novel method. Next adapted scorer must have
own specification, manual count/curve/tie/ID/schema/error fixtures before use.
