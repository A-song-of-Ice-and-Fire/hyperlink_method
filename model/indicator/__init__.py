from model.indicator.DE import Density
from model.indicator.TRW import TargetRandomWalk
from .VVAA import VertVertAdamicAdar
from .VEAA import VertEdgeAdamicAdar
from .EVAA import EdgeVertAdamicAdar
from .EEAA  import EdgeEdgeAdamicAdar
from .CLAA import CliqueLineAdamicAdar
from .CSAA import CliqueStarAdamicAdar
from .CN import CommonNeighbor
from .CE import CoexistEdge
from .CO import CommonObject
from .RA import ResourceAllocation
from .JC import Jaccard
from .COS import CosineSimilarity
from .WK2 import Walk2
from .WK3 import Walk3
from .HKatz import HyperKatzIndex
from .SKatz import SimpleKatzIndex
from .HWalk import HyperWalk
from .HEffi import HyperEfficiency
from .RWR import RestartRandomWalk
from .SHEffi import SuperposedHyperEfficiency
from .LTRW import LocalTargetRandomWalk
from .SLTRW import SuperposedLocalTargetRandomWalk
from .StRA import StableResourceAllocation
from .SuRA import SuperposeResourceAllocation
from .SRA import SimpleResourceAllocation
from .PS import ProbabilisticSpread
from .SPS import SimpleProbabilisticSpread
from .SC import SpreadComb
from .SSC import SuperposeSpreadComb
from .LRW import LocalRandomWalk
indicator_abb_map = {
    "VVAA" : VertVertAdamicAdar,
    "VEAA" : VertEdgeAdamicAdar,
    "EVAA" : EdgeVertAdamicAdar,
    "EEAA" : EdgeEdgeAdamicAdar,
    "CLAA" : CliqueLineAdamicAdar,
    "CSAA" : CliqueStarAdamicAdar,
    "CN"   : CommonNeighbor,
    "CE"   : CoexistEdge,
    "CO"   : CommonObject,
    "RA"    : ResourceAllocation,
    "JC"    : Jaccard,
    "COS"   : CosineSimilarity,
    "DE"    : Density,
    "WK2"   : Walk2,
    "WK3"   : Walk3,
    "HKatz" : HyperKatzIndex,
    "SKatz" : SimpleKatzIndex,
    "HWalk" : HyperWalk,
    "HEffi" : HyperEfficiency,
    "TRW"   : TargetRandomWalk,
    "RWR"   : RestartRandomWalk,
    "LTRW"  : LocalTargetRandomWalk,
    "SHEffi"    : SuperposedHyperEfficiency,
    "SLTRW"     : SuperposedLocalTargetRandomWalk,
    "SRA"     : SimpleResourceAllocation,
    "StRA"    : StableResourceAllocation,
    "SuRA"   : SuperposeResourceAllocation,
    "PS"      : ProbabilisticSpread,
    "SPS"     : SimpleProbabilisticSpread,
    "SC"    : SpreadComb,
    "SSC"   : SuperposeSpreadComb
}
from .ILRRA import InvLRResourceAllocation

indicator_abb_map["ILRRA"] = InvLRResourceAllocation