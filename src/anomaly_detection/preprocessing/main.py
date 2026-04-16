#from .preprocessing import *



from anomaly_detection.preprocessing.pipeline import PreprocessingPipeline
from anomaly_detection.preprocessing.transforms.scaling import ScalerTransform
from anomaly_detection.preprocessing.transforms.feature_selection import FeatureSelector


from sklearn.preprocessing import StandardScaler




from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[3]  # __file__ -> actual file location
# main.py → data → anomaly_detection → src → project root

TRAIN_PATH = BASE_DIR / "data" / "servers" / "X_part2.npy"
VAL_PATH = BASE_DIR / "data" / "servers" / "X_val_part2.npy"
Y_VAL_PATH = BASE_DIR / "data" / "servers" / "y_val_part2.npy"


from anomaly_detection.data.data import DataModule








if __name__=="__main__":


	"""
	prep = PreprocessingPipeline([
    	ScalerTransform(StandardScaler)
	])

	print(prep)
	"""

	
	prep = PreprocessingPipeline([
	    ScalerTransform(StandardScaler),
	    FeatureSelector([0,1,2,3,4,5])
	])

	#prep = PreprocessingPipeline([])
	


	# get data
	data = DataModule(TRAIN_PATH, VAL_PATH, Y_VAL_PATH)
	X_train, X_val, y_val = data.load()

	# 
	X_train_prep, X_val_prep = prep.fit_transform_split(X_train, X_val)
	print(X_train_prep[:3])

	#
	a = prep.get_artifacts()
	print(a)






# python -m anomaly_detection.preprocessing.main