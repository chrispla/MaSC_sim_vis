''' 
Script to compute selected features from audio, 
cross-refence song title with unique tag, 
and output csv with coordinates for VR
'''

# Imports
import csv
import os
import sys
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Configuration
csv_path = "/Users/chris/Google Drive/Work and Research/MaSC/PPTP/Arab Mashriq/corpus_statistics.csv" #Change path of tags csv here
output_path = "/Users/chris/Google Drive/Work and Research/MaSC/PPTP/Arab Mashriq/coordinates.csv" #Change path of output csv here
feature = input("Enter desired feature (mfcc, chromagram, mel_pca): ")
cluster_no = input("Enter desired number of clusters: ")

# Load .csv with unique tag references

with open(csv_path, "r") as f:
    reader = csv.reader(f)
    data = list(reader)

# Dictionary for referencing tag with true filepath
tags = {}

# File reading
all_paths = []
for root, dirs, files in os.walk('.'): #Change audio file directory here
        for name in files:
            if '.wav' in name:
                filepath = os.path.join(root, name)
                #search if name is substring of directory in csv
                for row in data: #this loop is inneficient
                    if name in row[1]: 
                        tags[filepath] = row[0] #be able to retrieve tag from current path
                        all_paths.append(filepath)
                        sys.stdout.write("\rLoading %i recordings." % (len(all_paths)))
                        sys.stdout.flush()
file_no = len(all_paths)
print() #newline

# Compute features
if feature == 'mfcc':

    X = []
    for i in range(file_no):

        #Load file
        y, sr = librosa.core.load(all_paths[i], duration=15.)
        #Features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        X.append(mfcc.flatten())

        sys.stdout.write("\rComputed MFCC for %i recordings." % (i))
        sys.stdout.flush()

    print()

elif feature == 'chromagram':

    X = []
    for i in range(file_no):
        
        #Load file
        y, sr = librosa.core.load(all_paths[i], duration=15.)
        #Features
        hop_length = 512
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        X.append(chroma)

        sys.stdout.write("\rComputed chromagrams for %i recordings." % (i))
        sys.stdout.flush()

    print()

elif feature == 'mel_pca':

    mel = []
    for i in range(file_no):

        #Load file
        y, sr = librosa.core.load(all_paths[i])
        #Features
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel.append(mel.flatten())

    sys.stdout.write("\rComputed mel spectrograms for %i recordings." % (i))
    sys.stdout.flush()

    X = PCA(n_components=20).fit_transform(mel) #Change number of principal components here

    print("\nAnalyzed principal components.")
    

else:
    print("Feature entered is invalid.")
    exit()

# Standardize
print("Standardizing...")
scl = StandardScaler()
X_s = scl.fit_transform(X)

# t-SNE
print("Reducing dimensionality...")
X_t = TSNE(n_components=2).fit_transform(X_s)

# Kmeans
print("Clustering...")
X_c = KMeans(n_clusters=cluster_no).fit_transform(X_t)

# Output
output_data = []
output_data.append(['Name', 'f1', 'f2', 'f3', 'color'],)

for i in range(file_no):
    output_data.append([]) #desired formatting has empty row between entries
    output_data.append(tags[all_paths[i]], X_t[i][0], X_t[i][1], X_t[i][2], X_c[i])

with open(output_path, "w") as f:
    writer = csv.writer(f)
    writer.writerows(output_data)