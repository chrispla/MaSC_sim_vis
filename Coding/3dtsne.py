import glob
import numpy as np
import librosa
import os
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import librosa.feature

#File reading
all_dirs = []
for root, dirs, files in os.walk('.'):
        for name in files:
            if '.wav' in name:
                filedir = os.path.join(root, name)
                all_dirs.append(filedir)

#Feature extraction
all_chroma = np.zeros(len(all_dirs))

for i in range(len(all_dirs)):

    #Section of file to be analyzed

    #duration = librosa.get_duration(filename=all_dirs[i]) // Loading with filename does not seem to be working
    #so I have to load the file twice, once to get its duration and once to load its appropriate section
    y, sr = librosa.core.load(all_dirs[i])
    duration = librosa.get_duration(y=y, sr=sr)
    offset = (duration/2) - 5

    #Load file
    y, sr = librosa.core.load(all_dirs[i], sr=22050, mono=True, offset=offset, duration=5)

    #Compute Constant-Q Chromagram
    hop_length = 512
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    all_chroma[i] = chroma

#Standardization
scaler = StandardScaler()
all_chroma_scaled = scaler.fit_transform(all_chroma)

#Dimensionality Reduction
all_chroma_scaled_red2 = TSNE(n_components=2, perplexity = 10.0).fit_transform(all_chroma_scaled)
all_chroma_scaled_red3 = TSNE(n_components=3, perplexity = 10.0).fit_transform(all_chroma_scaled)

#KMeans
chroma2_clusters = KMeans(n_clusters=5).fit_transform(all_chroma_scaled_red2)
chroma3_clusters = KMeans(n_clusters=5).fit_transform(all_chroma_scaled_red3)

#Plotting
fig = pyplot.figure()
plt.subplot(221)
plt.scatter(all_chroma_scaled_red2[:, 0], all_chroma_scaled_red2[:, 1], c=chroma2_clusters)

ax = Axes3D(fig)
ax.scatter(all_chroma_scaled_red3[:, 0], all_chroma_scaled_red3[:, 1], all_chroma_scaled_red3[:, 2], c=chroma3_clusters)
