import glob
import numpy as np
import librosa
import os
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import vega
import altair as alt
import pandas as pd
import scipy.signal
import scipy.fftpack as fft
from keras.layers import Input, Dense
from keras.models import Model

#File reading
all_dirs = []
for root, dirs, files in os.walk('./Test0'):
        for name in files:
            if '.wav' in name:
                filedir = os.path.join(root, name)
                all_dirs.append(filedir)

#Feature Computation
all_mel = []
file_names = []
file_no = len(all_dirs)
print(file_no)
for i in range(file_no):
    if (librosa.get_duration(filename=all_dirs[i]) > 13.):
        file_names.append('file:///' + str(all_dirs[i]))
        #Progress report
        if (i==file_no-1):
            print('100%')
        elif (i==int(file_no*0.75)):
            print('75%')
        elif (i==int(file_no*0.5)):
            print('50%')
        elif (i==int(file_no*0.25)):
            print('25%')
        #Load file
        y, sr = librosa.core.load(all_dirs[i], duration=13.)
        #Features
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        all_mel.append(mel.flatten()) #size (71680,)

#Principal Component Analysis
all_mel_pca = PCA(n_components=10).fit_transform(all_mel)

print(np.asarray(all_mel).shape)
print(all_mel_pca.shape)

#Standardization
scl1 = StandardScaler()
all_mel_pca_scaled = scl1.fit_transform(all_mel_pca)

#TSNE
all_mel_pca_scaled_red2 = TSNE(n_components=2).fit_transform(all_mel_pca_scaled)

#KMeans
kmeans = KMeans(n_clusters=5, random_state=0).fit(all_mel_pca_scaled_red2)
clusters = kmeans.predict(all_mel_pca_scaled_red2)

all_db = []
for i in range(file_no):
    if (librosa.get_duration(filename=all_dirs[i]) > 13.):
        #Progress report
        if (i==file_no-1):
            print('100%')
        elif (i==int(file_no*0.75)):
            print('75%')
        elif (i==int(file_no*0.5)):
            print('50%')
        elif (i==int(file_no*0.25)):
            print('25%')
        #Load file
        y, sr = librosa.core.load(all_dirs[i], duration=13.)
        #Features
        stft = librosa.core.stft(y=y)
        db = librosa.core.power_to_db(stft)
        all_db.append(np.mean(db)) #size (71680,)

#Standardization
scl2 = StandardScaler()
all_db_scaled = scl1.fit_transform((np.asarray(all_db)).reshape(-1, 1))
all_db_scaled = all_db_scaled.reshape(1,-1)

#x and y
mel1 = []
mel2 = []
for i in range(len(all_mel_pca_scaled_red2)):
    mel1.append(all_mel_pca_scaled_red2[i][0])
    mel2.append(all_mel_pca_scaled_red2[i][1])

#Visualization
df = pd.DataFrame({'x': mel1, 'y': mel2, 'color': clusters, 'path': np.asarray(file_names), 'filename': np.asarray(file_names)})
chart = alt.Chart(df).mark_circle(opacity=0.6, size=60).encode(x='x', y='y', color='color:N', href='path', tooltip=['filename']).interactive()
display(chart)
