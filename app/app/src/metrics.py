from sklearn.metrics import accuracy_score

def compute_far_frr(labels,predictions):

    false_accept=0
    false_reject=0
    genuine=0
    imposter=0

    for t,p in zip(labels,predictions):

        if t==1:

            genuine+=1

            if p==0:
                false_reject+=1

        else:

            imposter+=1

            if p==1:
                false_accept+=1

    FAR=false_accept/imposter if imposter else 0
    FRR=false_reject/genuine if genuine else 0

    return FAR,FRR


def compute_accuracy(labels,predictions):

    return accuracy_score(labels,predictions)