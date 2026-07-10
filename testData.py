from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
from tempfile import TemporaryFile
import os
import pickle
import random 
import operator
import math
import numpy as np
from collections import defaultdict

dataset = []

def loadDataset(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break

loadDataset("my.dat")

# Use the same distance function as in music_genre.py for consistency
def calculate_distance(instance1, instance2, k):
    # Extract mean and covariance from the feature tuples
    mean1, cov1, _ = instance1
    mean2, cov2, _ = instance2
    
    # Calculate distance between means
    mean_distance = np.linalg.norm(mean1 - mean2)
    
    # Calculate distance between covariances (using Frobenius norm)
    cov_distance = np.linalg.norm(cov1 - cov2, ord='fro')
    
    # Combine the distances (you can adjust weights if needed)
    total_distance = mean_distance + cov_distance
    
    return total_distance

def getNeighbors(trainingSet , instance , k):
    distance_list = []
    for x in range(len(trainingSet)):
        dist = calculate_distance(trainingSet[x], instance, k)
        distance_list.append((trainingSet[x][2], dist))
    distance_list.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distance_list[x][0])
    return neighbors  

def nearestClass(neighbors):
    classVote ={}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response]+=1 
        else:
            classVote[response]=1 
    sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)
    return sorter[0][0]

# Create proper genre mapping based on the training data
genre_mapping = {
    1: "blues",
    2: "classical", 
    3: "country",
    4: "disco",
    5: "hiphop",
    6: "jazz",
    7: "metal",
    8: "pop",
    9: "reggae",
    10: "rock"
}

(rate,sig)=wav.read("test_music/metal-dark-matter-111451.wav")
mfcc_feat=mfcc(sig,rate,winlen=0.020,appendEnergy=False)
covariance = np.cov(np.matrix.transpose(mfcc_feat))
mean_matrix = mfcc_feat.mean(0)
feature=(mean_matrix,covariance,0)

pred=nearestClass(getNeighbors(dataset ,feature , 5))

print(f"Predicted genre: {genre_mapping[pred]}")
print(f"Predicted class number: {pred}")