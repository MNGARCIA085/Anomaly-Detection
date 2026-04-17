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





import pickle

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)



def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)





import mlflow
import pickle
import os

# load form a mlflow run id
def load_pipeline_from_run(run_id, artifact_path="pipeline.pkl"):
    # downloads artifact locally and returns the local path
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_path
    )

    with open(local_path, "rb") as f:
        return pickle.load(f)






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



	# test logging


	from anomaly_detection.infra.conf import init_mlflow


	init_mlflow('Anomaly_Detection')


	import pickle
	import mlflow

	with open("pipeline.pkl", "wb") as f:
	    pickle.dump(prep, f)


	"""
	mlflow.log_artifact("pipeline.pkl")
	mlflow.log_dict(prep.get_artifacts(), "artifacts.json")  # for inspection
	"""

	#
	run_id = "23fbbb8d7fce4b6cbe4de1f35215ea67"

	pipeline = load_pipeline_from_run(run_id)
	print(pipeline)

	d = pipeline.get_artifacts()

	print(d)

	s = d['step_0_ScalerTransform']
	print(s)
	print(s.mean_)

	#X_new = pipeline.transform(X_new)


	"""
	To make inference
		pipeline = load_pickle("pipeline.pkl")
		X_new = pipeline.transform(X_new)
	"""

	pipeline = load_pickle("pipeline.pkl")







# python -m anomaly_detection.preprocessing.main