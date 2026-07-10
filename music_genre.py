from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np

from tempfile import TemporaryFile
import os
import pickle
import random 
import operator

import math
from collections import defaultdict

# Function to calculate the Euclidean distance between two instances
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

def getNeighbours(trainingSet, instance, k):
    distance_list = []
    for x in range(len(trainingSet)):
        dist = calculate_distance(trainingSet[x], instance, k)
        distance_list.append((trainingSet[x][2], dist))
    distance_list.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distance_list[x][0])
    return neighbors

# Function to find the nearest class
def nearestClass(neighbors):
    classVote = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response]+=1 
        else:
            classVote[response]=1
    sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)
    return sorter[0][0]

def getAccuracy(testSet, predictions):
    correct = 0 
    for x in range (len(testSet)):
        if testSet[x][-1]==predictions[x]:
            correct+=1
    return 1.0*correct/len(testSet)

directory = "Data/genres_original/"
f= open("my.dat", 'wb')
i=0

print("Processing audio files...")
for folder in os.listdir(directory):
    i+=1
    if i==11 :
        break
    print(f"Processing folder {i}: {folder}")
    for file in os.listdir(directory+folder):
        if file.endswith('.wav'):  # Only process WAV files
            try:
                (rate, sig) = wav.read(directory+folder+"/"+file)
                mfcc_feat = mfcc(sig,rate, winlen=0.020, appendEnergy = False)
                covariance = np.cov(np.matrix.transpose(mfcc_feat))
                mean_matrix = mfcc_feat.mean(0)
                feature = (mean_matrix, covariance, i)
                pickle.dump(feature, f)
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
        
f.close()

dataset = []
def loadDataset(filename , split , trSet , teSet):
    with open("my.dat" , 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break  

    for x in range(len(dataset)):
        if random.random() <split :      
            trSet.append(dataset[x])
        else:
            teSet.append(dataset[x])  

trainingSet = []
testSet = []
loadDataset("my.dat" , 0.66, trainingSet, testSet)

print(f"Training set size: {len(trainingSet)}")
print(f"Test set size: {len(testSet)}")

leng = len(testSet)
predictions = []
for x in range (leng):
    predictions.append(nearestClass(getNeighbours(trainingSet ,testSet[x] , 5))) 

accuracy1 = getAccuracy(testSet , predictions)
print(f"Accuracy: {accuracy1}")