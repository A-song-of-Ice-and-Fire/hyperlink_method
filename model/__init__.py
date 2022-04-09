from model.module import Module
from .indicator import (
    indicator_abb_map,
    NodeNodeAdamicAdar,
    NodeEdgeAdamicAdar,
    EdgeNodeAdamicAdar,
    EdgeEdgeAdamicAdar,
    CliqueLineAdamicAdar,
    CliqueStarAdamicAdar,
    ResourceAllocation,
    CommonNeighbor,
    CoexistEdge,
    CommonObject,
    Jaccard,
    CosineSimilarity,
    Density,
    Walk2,
    Walk3,
    HyperWalk,
    HyperEfficiency,
    HyperKatzIndex,
    TargetRandomWalk,
    LocalTargetRandomWalk,
    RestartRandomWalk,
    SimpleKatzIndex,
    SuperposedHyperEfficiency,
    SuperposedLocalTargetRandomWalk,
    StableResourceAllocation,
    SimpleResourceAllocation,
    SuperposeResourceAllocation,
    InvLRResourceAllocation,
    ProbabilisticSpread,
    SimpleProbabilisticSpread,
    SpreadComb,
    SuperposeSpreadComb
)
from .integrator import (
    integrator_abb_map,
    LogisticRegression,
    RandomForest,
    LightGBM
)
from .intensifier import (
    intensifier_abb_map,
    CollaborativeFilter,
    WeightedCollaborativeFilter
)


str_to_object = {**indicator_abb_map,**integrator_abb_map,**intensifier_abb_map}

def object_to_str(object:Module):
    if isinstance(object,NodeNodeAdamicAdar):
        return "NNAA"
    elif isinstance(object,NodeEdgeAdamicAdar):
        return "NEAA"
    elif isinstance(object,EdgeNodeAdamicAdar):
        return "ENAA"
    elif isinstance(object,EdgeEdgeAdamicAdar):
        return "EEAA"
    elif isinstance(object,CliqueLineAdamicAdar):
        return "CLAA"
    elif isinstance(object,CliqueStarAdamicAdar):
        return "CSAA"
    elif isinstance(object,CoexistEdge):
        return "CE"
    elif isinstance(object,CommonNeighbor):
        return "CN"
    elif isinstance(object,CommonObject):
        return "CO"
    elif isinstance(object,CollaborativeFilter):
        return "CF"
    elif isinstance(object,WeightedCollaborativeFilter):
        return "WCF"
    elif isinstance(object,SimpleResourceAllocation):
        return "SRA"
    elif isinstance(object,SuperposeResourceAllocation):
        return "SuRA"    
    elif isinstance(object,StableResourceAllocation):
        return "StRA"
    elif isinstance(object,ResourceAllocation):
        return "RA"
    elif isinstance(object,Jaccard):
        return "JC"
    elif isinstance(object,CosineSimilarity):
        return "COS"
    elif isinstance(object,Walk2):
        return "WK2"
    elif isinstance(object,Walk3):
        return "WK3"
    elif isinstance(object,HyperKatzIndex):
        return "HKatz"
    elif isinstance(object,SimpleKatzIndex):
        return "SKatz"
    elif isinstance(object,Density):
        return "DE"
    elif isinstance(object,HyperWalk):
        return "HWalk"
    elif isinstance(object,HyperEfficiency):
        return "HEffi"
    elif isinstance(object,SuperposedHyperEfficiency):
        return "SHEffi"
    elif isinstance(object,LogisticRegression):
        return "LR"
    elif isinstance(object,RandomForest):
        return "RF"
    elif isinstance(object,LightGBM):
        return "LightGBM"
    elif isinstance(object,TargetRandomWalk):
        return "TRW"
    elif isinstance(object,RestartRandomWalk):
        return "RRW"
    elif isinstance(object,LocalTargetRandomWalk):
        return "LTRW"
    elif isinstance(object,SuperposedLocalTargetRandomWalk):
        return "SLTRW"
    elif isinstance(object,InvLRResourceAllocation):
        return "ILRRA"
    elif isinstance(object,ProbabilisticSpread):
        return "PS"
    elif isinstance(object,SimpleProbabilisticSpread):
        return "SPS"
    elif isinstance(object,SpreadComb):
        return "SC"
    elif isinstance(object,SuperposeSpreadComb):
        return "SSC"
    else:
        return ""