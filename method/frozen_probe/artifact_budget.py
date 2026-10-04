"""Prospective payload accounting; writers must separately enforce their bounds."""


def inspect_artifact_plan(array_specs, serialization_upper_bound_bytes,
                          metadata_upper_bound_bytes, snapshot_upper_bound_bytes,
                          log_report_upper_bound_bytes, ceiling_bytes):
    bounds = (serialization_upper_bound_bytes, metadata_upper_bound_bytes,
              snapshot_upper_bound_bytes, log_report_upper_bound_bytes, ceiling_bytes)
    if not all(type(x) is int and x >= 0 for x in bounds) or ceiling_bytes == 0:
        raise ValueError('Artifact bounds must be nonnegative integer byte counts')
    payloads = {}
    for name, spec in array_specs.items():
        shape, itemsize = spec['shape'], spec['itemsize']
        if not shape or not all(type(n) is int and n > 0 for n in shape) or type(itemsize) is not int or itemsize < 1:
            raise ValueError('Invalid array specification')
        size = itemsize
        for n in shape:
            size *= n
        payloads[name] = size
    array_bytes = sum(payloads.values())
    total = array_bytes + sum(bounds[:-1])
    return dict(array_payload_bytes=payloads, array_payload_total_bytes=array_bytes,
                bounded_other_bytes=sum(bounds[:-1]), estimated_upper_bound_bytes=total,
                ceiling_bytes=ceiling_bytes, remaining_bytes=ceiling_bytes-total,
                passed=total <= ceiling_bytes,
                limitation='Prospective plan only: serialization, metadata, snapshot and log/report bounds require writer enforcement; not peak RAM or actual disk measurement')
