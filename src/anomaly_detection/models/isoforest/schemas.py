from dataclasses import dataclass



from dataclasses import dataclass

@dataclass
class IsoForestConfig:
    n_estimators: int = 100
    contamination: float = 0.01 # carefull with this, i will alredy tune theshold
    random_state: int = 42 # more global later

