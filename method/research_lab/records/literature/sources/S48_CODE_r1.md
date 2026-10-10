# S48 code follow-up r1 — pinned OSCR function audit

2026-10-05. Original S48 selected paper card retained as historical read scope.
Official project-linked repo HSP-IIT/fsosar, pinned
[commit c8115c09c54d5480bf0adf972f66988bb2274c2b](https://github.com/HSP-IIT/fsosar/tree/c8115c09c54d5480bf0adf972f66988bb2274c2b).
Source `utils.py::compute_oscr` lines339–366 fully inspected; utils33525B/
SHA55b79e5e9ca57956b499427dec97b187b464faad7fb4d382e0c3263542738b6b.
`train.py` selected callsites inspected in D075 cycle: metric calls358–360/380.
Not fullrepo/model/training/data/paper-score reproduction. Public browser raw
cachemiss; direct HTTPS worked. Full root BSD3-Clause LICENSE1518B/SHA
d3522fc7814fc79c74f64c68dc9cb591b062b5e90a5bddd50e444e5b3f727711 retained with
source at method/frozen_probe/references/fsosar_c8115c09/; exact35043B capture.
No environment install or upstreammodule imported/executed; AST function only.

Pinned function denominator FP/(FP+Nunknown), forces terminalpoint(1,1).
OSCRREFTEST002 actual synthetic areas.25/1/.25 for allknownwrong/perfectseparated/
wrongtied; manual jointarea0/1/0. Two zero-CCR invariant disagreements reject this
specific scorer variant for H007. OSCRREFTEST001 own wrapper NumPyimport-hook
failure retained, not a scientific finding/reference algorithm result.
Protocol versions distinguish failedexecution, completedaudit and failedparity.

This finding does NOT establish published FR-Disc numbers are wrong: actual
paper result-generating sourceversion/predictions unknown. No author contact,
issue posting, numerical paper reproduction, broad method-invalidity claim.
Original OSCR denominator/endpoint primary S50; our inclusive groupedtrapezoid
scorer is explicitly adapted, not an exact official scorer or novel metric.
Follow-up: strong FR-Disc method comparison needs train/supervision/episode/input
contracts and released-result provenance, not reusing a rejected metric function.
