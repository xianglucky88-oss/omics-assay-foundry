from .controls import prepare
from .evidence import wilson
from ..validation import number

def process(payload):
    rows = prepare(payload)
    z = number(payload.get("confidence_z", 1.959963984540054), "confidence_z", minimum=0.1, maximum=10)
    groups = {}
    sites = []
    for name, marked, total, region in rows:
        sites.append({"id": name, "region": region, "methylated": marked, "coverage": total, **wilson(marked,total,z)})
        previous = groups.setdefault(region, [0,0,0])
        previous[0] += marked
        previous[1] += total
        previous[2] += 1
    regions = [{"id": name, "methylated": values[0], "coverage": values[1], "sites": values[2],
                **wilson(values[0], values[1], z)} for name,values in sorted(groups.items())]
    return {"model": "independent-homogeneous-binomial-zero-conversion-error", "interval": "Wilson score",
            "confidence_z": z, "sites": sites, "regions": regions}
