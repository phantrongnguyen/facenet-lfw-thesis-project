import time
import numpy as np

from tqdm import tqdm

from .embedding_extractor import get_embedding
from .similarity import cosine_similarity
from .metrics import compute_far_frr
from .metrics import compute_accuracy


def evaluate(model,pairs,threshold):

    similarities=[]
    labels=[]
    predictions=[]
    times=[]

    for img1,img2,label in tqdm(pairs):

        start=time.time()

        emb1=get_embedding(img1,model)
        emb2=get_embedding(img2,model)

        sim=cosine_similarity(emb1,emb2)

        end=time.time()

        similarities.append(sim)
        labels.append(label)

        pred=1 if sim>threshold else 0

        predictions.append(pred)

        times.append(end-start)

    acc=compute_accuracy(labels,predictions)

    FAR,FRR=compute_far_frr(labels,predictions)

    avg_time=np.mean(times)

    return similarities,labels,acc,FAR,FRR,avg_time