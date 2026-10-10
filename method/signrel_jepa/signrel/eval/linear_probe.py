import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,balanced_accuracy_score


def linear_probe(train_x,train_y,test_x,test_y,seed=42,c=1.,max_iter=1000):
    train_x,test_x = np.asarray(train_x),np.asarray(test_x)
    train_y,test_y = np.asarray(train_y,dtype=str),np.asarray(test_y,dtype=str)
    if len(set(train_y))<2 or len(test_y)==0:
        raise ValueError("Probe needs >=2 training classes and a nonempty evaluation set")
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(C=c,max_iter=max_iter,random_state=seed,solver="lbfgs")
    model.fit(scaler.transform(train_x),train_y)
    probs = model.predict_proba(scaler.transform(test_x))
    order = np.argsort(-probs,axis=1,kind="stable")
    pred = model.classes_[order[:,0]]
    top5 = np.array([y in model.classes_[idx[:5]] for y,idx in zip(test_y,order)])
    covered = np.isin(test_y,model.classes_)
    majority = sorted(set(train_y),key=lambda y:(-(train_y==y).sum(),y))[0]
    return {"top1":float((pred==test_y).mean()),"top5":float(top5.mean()),
            "macro_f1":float(f1_score(test_y,pred,average="macro",zero_division=0)),
            "balanced_accuracy":float(balanced_accuracy_score(test_y,pred)),
            "per_class_accuracy":{y:float((pred[test_y==y]==y).mean()) for y in sorted(set(test_y))},
            "majority_baseline":float((test_y==majority).mean()),"train_samples":len(train_y),
            "test_samples":len(test_y),"train_classes":len(model.classes_),"known_class_coverage":float(covered.mean()),
            "seed":seed,"C":c,"normalization":"standardization fit to train only"}
