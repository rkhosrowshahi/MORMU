import numpy as np
import torch
import torch.nn.functional as F
# from sklearn.cluster import KMeans
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_mia(shadow_model, target_model, shadow_train_loader, shadow_unseen_loader, forget_loader):

    in_logits, out_logits, forget_logits = [], [], []
    with torch.no_grad():
        for i in range(shadow_train_loader.num_batches):
            X, y = shadow_train_loader.next(i)
            X, y = X.to(device), y.to(device)
            out = shadow_model(X)
            out = F.softmax(out, dim=1)
            in_logits.extend(out.cpu().numpy())
            # retain_logits.extend(nn.functional.softmax(out, dim=1).cpu().numpy())
        for i in range(shadow_unseen_loader.num_batches):
            X, y = shadow_unseen_loader.next(i)
            X, y = X.to(device), y.to(device)
            out = shadow_model(X)
            out = F.softmax(out, dim=1)
            out_logits.extend(out.cpu().numpy())
            # test_logits.extend(nn.functional.softmax(out, dim=1).cpu().numpy())
            
            # print(nn.functional.softmax(out, dim=1))
        for i in range(forget_loader.num_batches):
            X, y = forget_loader.next(i)
            X, y = X.to(device), y.to(device)
            out = target_model(X)
            out = F.softmax(out, dim=1)
            forget_logits.extend(out.cpu().numpy())
            # forget_logits.extend(nn.functional.softmax(out, dim=1).cpu().numpy())

    X_train = np.vstack([in_logits, out_logits])

    y_train = np.hstack([np.zeros(len(in_logits)), np.ones(len(out_logits))])

    attacker = MLPClassifier(hidden_layer_sizes=(64), max_iter=1000, batch_size=32, random_state=0)

    attacker.fit(X_train, y_train)

    y_pred = attacker.predict(forget_logits)
    tpr = 0
    tn, fp, fn, tp = confusion_matrix(y_true=np.ones(len(forget_logits)), y_pred=y_pred).ravel()

    tpr = tp / (tp+fn)

    return tpr
    