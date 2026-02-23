from dataclasses import dataclass

@dataclass
class EvalWeights:
    vp_weight: float = 10.0
    settlement_weight: float = 2.0
    city_weight: float = 3.0
    road_weight: float = 1.0
    resource_weight: float = 0.5



    
        
        
        
       