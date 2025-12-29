# CDT Models
# v1: Pooled embeddings (POC)
# v2: Sequence-level embeddings (interpretable attention)

from .cdt_poc_model import CDTPOCModel
from .cdt_v2_seqlevel import CDTv2Model, CDTv2Config

__all__ = [
    # v1 (POC)
    "CDTPOCModel",
    # v2 (Sequence-Level)
    "CDTv2Model",
    "CDTv2Config",
]
