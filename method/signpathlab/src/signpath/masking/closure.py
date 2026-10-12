from ..contracts import MaskPlan


class ClosureResolver:
    def resolve(self, token_bundle, raw_mask, *, geometry_eligibility="fully_hidden"):
        if geometry_eligibility not in {"fully_hidden", "any_overlap"}:
            raise ValueError("Unknown geometry eligibility")
        if raw_mask.regime == "strict_temporal" and geometry_eligibility != "fully_hidden":
            raise ValueError("Strict temporal geometry targets must be fully hidden")
        cells = raw_mask.cells
        visible, hidden, targets, geo_visible, geo_targets = [], [], [], [], []
        per_stream = {str(s): {"visible": 0, "hidden": 0, "targets": 0} for s in range(7)}
        for index, token in enumerate(token_bundle.tokens):
            if not token.valid:
                continue
            touched = bool(token.raw_support & cells)
            (hidden if touched else visible).append(index)
            per_stream[str(token.stream)]["hidden" if touched else "visible"] += 1
            if touched and token.target_valid:
                targets.append(index)
                per_stream[str(token.stream)]["targets"] += 1
        geo_counts = {}
        for index, token in enumerate(token_bundle.geometry):
            if not token.valid:
                continue
            counts = geo_counts.setdefault(f"{token.group}/{token.scale}", {"visible": 0, "targets": 0})
            if not token.support & cells:
                geo_visible.append(index)
                counts["visible"] += 1
            elif geometry_eligibility == "any_overlap" or token.support <= cells:
                geo_targets.append(index)
                counts["targets"] += 1
        total = len(visible) + len(hidden)
        diagnostics = {"mask_ratio_requested": raw_mask.requested_ratio,
                       "mask_regime": raw_mask.regime, "mask_strategy": raw_mask.strategy, "anatomical_region": raw_mask.region,
                       "raw_source_mask_ratio_actual": raw_mask.actual_source_ratio,
                       "token_mask_ratio_after_closure": len(hidden) / total if total else 0.0,
                       "visible_raw_tokens": len(visible), "semantic_targets": len(targets),
                       "geometry_targets": len(geo_targets), "per_stream": per_stream, "per_group_scale": geo_counts}
        return MaskPlan(tuple(visible), tuple(hidden), tuple(targets), tuple(geo_visible), tuple(geo_targets), raw_mask, diagnostics)
