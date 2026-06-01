import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def plot_roc(labels,similarities,model):

    fpr,tpr,_=roc_curve(labels,similarities)

    roc_auc=auc(fpr,tpr)

    plt.plot(fpr,tpr,label=f"{model} AUC={roc_auc:.3f}")