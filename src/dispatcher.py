from sklearn import ensemble
from sklearn import linear_model

MODELS = {
    "randomforest":ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "logisticregression":linear_model.LogisticRegression(C=0.12,solver = 'lbfgs',n_jobs=-1,verbose=2,max_iter = 200),
    "extratrees":ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2),
}