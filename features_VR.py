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
audio_dir = "/Volumes/Extreme SSD/middle15" #audio file directory
output_dir = "/Users/chris/Google Drive/Work and Research/MaSC/PPTP/Scripts/" #Change path of output csv here
feature = input("Enter desired feature (mfcc, chromagram, mel_pca): ")
#cluster_no = input("Enter desired number of clusters: ")
cluster_no = 6

# File reading
all_paths = []
all_names = []
for root, dirs, files in os.walk(audio_dir): #Change audio file directory here
    for name in files:
        if '.wav' in name:
            filepath = os.path.join(root, name)
            all_paths.append(filepath)
            all_names.append(name[:-4])
        if (len(all_paths) > 20):
            break

file_no = len(all_paths)
print("Number of files:", file_no)

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

    X = []
    for i in range(file_no):

        #Load file
        y, sr = librosa.core.load(all_paths[i])
        #Features
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        X.append(mel.flatten())

        sys.stdout.write("\rComputed mel spectrograms for %i recordings." % (i))
        sys.stdout.flush()

    X = PCA(n_components=20).fit_transform(X) #Change number of principal components here

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
X_t = TSNE(n_components=3).fit_transform(X_s)

# Kmeans
print("Clustering...")
X_c = KMeans(n_clusters=cluster_no).fit(X_t)
clusters = X_c.predict(X_t)

# Output
output_data = []
output_data.append(['Name', 'f1', 'f2', 'f3', 'color'],)

colors = ['red', 'skyblue', 'goldenrod', 'darkred', 'darkblue', 'olive']
for i in range(file_no):
    output_data.append([]) #desired formatting has empty row between entries
    output_data.append([all_names[i], X_t[i][0], X_t[i][1], X_t[i][2], colors[clusters[i]]])

with open(output_dir + 'features.csv', "w") as f:
    writer = csv.writer(f)
    writer.writerows(output_data)