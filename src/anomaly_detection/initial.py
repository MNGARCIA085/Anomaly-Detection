


#-----------DATA MODULE------------------#
class DataModule:

    def __init__(self, train_path, val_path, y_val_path):
        self.train_path = train_path
        self.val_path = val_path
        self.y_val_path = y_val_path

    def load(self):
        X_train = np.load(self.train_path)
        X_val = np.load(self.val_path)
        y_val = np.load(self.y_val_path)
        return X_train, X_val, y_val








# ----------------- PREPROCESSING ------------------------#


# base
class Transform(ABC):

    def fit(self, X):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# reusable pieces
class ScalerTransform(Transform):

    def __init__(self, scaler_cls):
        self.scaler = scaler_cls()

    def fit(self, X):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)


    def get_artifact(self):
    	return self.scaler




class FeatureSelector(Transform):

    def __init__(self, indices):
        self.indices = indices

    def transform(self, X):
        return X[:, self.indices]


    # EACH class defines whats worth saving
    def get_artifact(self):
       return {"indices": self.indices}


"""
later
PCA
Log transform
Clipping
Custom domain transform
"""



class PreprocessingPipelineV0:

    def __init__(self, steps):
        self.steps = steps  # list of Transform

    def fit(self, X):
        for step in self.steps:
            X = step.fit_transform(X)
        return self

    def transform(self, X):
        for step in self.steps:
            X = step.transform(X)
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def fit_transform_split(self, X_train, X_val):
        X_train_p = self.fit_transform(X_train)
        X_val_p = self.transform(X_val)
        return X_train_p, X_val_p





class PreprocessingPipeline:

    def __init__(self, steps):
        self.steps = steps

    def fit(self, X):
        for step in self.steps:
            X = step.fit_transform(X)
        return self

    def transform(self, X):
        for step in self.steps:
            X = step.transform(X)
        return X

    def fit_transform_split(self, X_train, X_val):
        self.fit(X_train)
        return self.transform(X_train), self.transform(X_val)

    def get_artifacts(self):
        artifacts = {}
        for i, step in enumerate(self.steps):
            if hasattr(step, "get_artifact"):
                artifacts[f"step_{i}_{step.__class__.__name__}"] = step.get_artifact()
        return artifacts










"""
examples

prep = PreprocessingPipeline([
    ScalerTransform(StandardScaler)
])

prep = PreprocessingPipeline([
    ScalerTransform(StandardScaler),
    FeatureSelector([0,1,2,3,4,5])
])

prep = PreprocessingPipeline([])
"""


"""
later: add builder

def build_preprocessor(cfg):

    steps = []

    if cfg.use_scaler:
        steps.append(ScalerTransform(cfg.scaler_cls))

    if cfg.feature_indices:
        steps.append(FeatureSelector(cfg.feature_indices))

    return PreprocessingPipeline(steps)

"""










# ----------------- BASE (asbtract class) ------------------------#


class AnomalyModel(ABC):
    @abstractmethod
    def fit(self, X_train, X_val=None): 
        pass


  
    @abstractmethod
    def get_scores(self, X): 
    	""" return scores """
        pass




#---------------------THRESHOLD----------------------#
class ThresholdSelector:

    @staticmethod
    def from_pr_curve(y, scores):
        precision, recall, thresholds = precision_recall_curve(y, scores)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return thresholds[np.argmax(f1)]



#---------------------EVALUATOR----------------------#
class Evaluator:

    def evaluate(self, y_true, scores, threshold):
        y_pred = scores > threshold
        return {
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
        }




# ----------------- ISOLATION FOREST ------------------------#
@dataclass
class IsoForestConfig:
    n_estimators=100, # not hardcode later; pass them all in config (hydra, but lets use this defaults)
	contamination=0.01,
	random_state=42


def build_forest(cfg: IsoForestConfig):
	return IsolationForest(
	    	n_estimators=cfg.n_estimators,
	    	contamination=cfg.contamination,
	    	random_state=cfg.random_state # later -> more global
	)


class IsolationForestModel(AnomalyModel):

	def __init__(self, model_cfg):
		self.iso = build_forest(model_cfg)

	def fit(self, X_train):
		self.iso.fit(X_train)



	def get_scores(self, X):
    	scores = self.iso.decision_function(X)  # higher = more normal
    	return -scores  # higher = more anomalous






# ----------------- AUTOENCODER ------------------------#
# dataclass for Model
@dataclass
class AEConfig:
    input_dim : 11


# Model
class AE(nn.Module):

    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(cfg.input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))



# Trainer
class AETrainer():


	# for now very simpel, later, also rain with x_val
	def train(self, model, X_train, X_val): # pass model here, not need to save state
		optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
		criterion = nn.MSELoss()

		# ---- Train (only normal data) ----
		X_t = torch.tensor(X_train, dtype=torch.float32)

		for epoch in range(50):
		    optimizer.zero_grad()
		    recon = model(X_t)
		    loss = criterion(recon, X_t)
		    loss.backward()
		    optimizer.step()

		return model



# wrapper
class AutoencoderModel(AnomalyModel):
    
    def __init__(self, model_cfg:AEConfig):
        self.model = AE(model_cfg)
        self.trainer = AETrainer() # Only NNs see the Trainer, later pass training_cfg

    def fit(self, X_train_prep, X_val_prep=None):
        self.trainer.train(self.model, X_train_prep, X_val_prep)


    def get_scores(self, X):
	    X_t = torch.tensor(X, dtype=torch.float32)
	    recon = self.model(X_t).detach()
	    error = ((X_t - recon) ** 2).mean(dim=1).numpy()
	    return error






#------------------------EXERPIMENT-------------------------#
class Experiment:

    def __init__(self, model, preprocessor, threshold_selector, evaluator):
        self.model = model
        self.preprocessor = preprocessor
        self.threshold_selector = threshold_selector
        self.evaluator = evaluator

    def run(self, X_train, X_val, y_val):

        X_train_p, X_val_p = self.preprocessor.fit_transform_split(X_train, X_val)


        # for dynamic inut dim
        # Determine input_dim dynamically from the preprocessed data
    	self.model.initialize(input_dim=X_train_p.shape[1])


        self.model.fit(X_train_p, X_val_p)

        scores_val = self.model.get_scores(X_val_p)

        threshold = self.threshold_selector.from_pr_curve(y_val, scores_val)

        metrics = self.evaluator.evaluate(y_val, scores_val, threshold)


        # 🔥 log artifacts
        if self.logger:
            artifacts = self.preprocessor.get_artifacts()
            self.logger.log_artifacts(artifacts)

        return metrics, threshold




#--------MLFLOW logging-----------#
class MLflowLogger:

    def log_artifacts(self, artifacts):
        for name, obj in artifacts.items():
            joblib.dump(obj, f"{name}.pkl")
            mlflow.log_artifact(f"{name}.pkl")


# https://gemini.google.com/app/36bc81e5d0ffd6c7






# registry....



"""
exp = Experiment(
    model=IsolationForestModel(cfg),
    preprocessor=StandardScalerPrep(),
    threshold_selector=ThresholdSelector(),
    evaluator=Evaluator()
)

exp.run(...)


swap only:
model = AutoencoderModel(...)

"""



"""
DataModule
   ↓
PreprocessingPipeline (with artifacts)
   ↓
Model
   ↓
ThresholdSelector
   ↓
Evaluator
   ↓
Experiment (logs everything)
"""