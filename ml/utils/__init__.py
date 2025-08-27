from typing import Any
from datetime import datetime, timezone
from .inspect_model import InspectModel

def get_timestamp() -> str:
    """
      Return UTC timestamp in the format yyyymmddHHMMSS
      Example:
      May 5th 2024 6:52:12 UTC time --> 20240509065212
    """
    date = datetime.now(timezone.utc)
    return datetime.strftime(date, "%Y%m%d%H%M%S")

def histogram(y:Any) -> dict:
    bins = [(-1,-0.5), (-0.5, 0.5), (0.5, 0.99), (0.99, 1.1)]
    result = [0] * len(bins)

    for i, (min, max) in enumerate(bins):
        result[i] = sum(1 for v in y if min <= v < max)
    result = [r/sum(result) for r in result]

    return {f"{bins[i][0]} <= y < {bins[i][1]}": result[i] for i in range(len(result))}
        