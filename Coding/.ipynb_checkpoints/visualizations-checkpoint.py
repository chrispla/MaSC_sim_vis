%matplotlib inline
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#File reading
all_dirs = []
for root, dirs, files in os.walk('./Test'):
        for name in files:
            if '.wav' in name:
                filedir = os.path.join(root, name)
                all_dirs.append(filedir)

all_chroma = []
all_mfcc = []
file_no = len(all_dirs)
print(file_no)
for i in range(file_no):
    print(str(i+1) + '/' + str(file_no))
    #Load file
    y, sr = librosa.core.load(all_dirs[i])
    #Features
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=48, n_octaves=7)
    all_mfcc.append(mfcc.flatten())
    all_chroma.append(chroma.flatten())

#Shift mffcs to positive values
#min = np.amin(all_mfcc)
#all_mfcc = all_mfcc + min

#Standardization
scl1 = StandardScaler()
all_mfcc_scaled = scl1.fit_transform(all_mfcc)
scl2 = StandardScaler()
all_chroma_scaled = scl2.fit_transform(all_chroma)

#Principal Component Analysis

#MFCC2
pca_mfcc2 = PCA(n_components=2).fit_transform(all_mfcc_scaled)
scl3 = StandardScaler()
pca_mfcc2 = scl3.fit_transform(pca_mfcc2)
pca_mfcc2x = []
pca_mfcc2y = []
for i in range(pca_mfcc2.shape[0]):
    pca_mfcc2x.append(pca_mfcc2[i][0])
for i in range(all_mfcc_scaled.shape[0]):
    pca_mfcc2y.append(pca_mfcc2[i][1])

#MFCC1
pca_mfcc = PCA(n_components=1).fit_transform(all_mfcc_scaled)
scl4 = StandardScaler()
pca_mfcc = scl4.fit_transform(pca_mfcc)
pca_mfcc1 = []
for i in range(pca_mfcc.shape[0]):
    pca_mfcc1.append(pca_mfcc[i][0])

#CHROMA1
pca_chroma = PCA(n_components=1).fit_transform(all_chroma_scaled)
scl5 = StandardScaler()
pca_chroma = scl5.fit_transform(pca_chroma)
pca_chroma1 = []
for i in range(pca_chroma.shape[0]):
    pca_chroma1.append(pca_chroma[i][0])

#Dimensionality reduction
#all_mfcc_scaled_red = TSNE(n_components=1, perplexity = 10.0).fit_transform(all_mfcc_scaled)
#all_chroma_scaled_red = TSNE(n_components=1, perplexity = 10.0).fit_transform(all_chroma_scaled)
#print('Computation complete.')


#Creating dataframe
#x = []
#y = []
#z = []
# for i in range(len(all_mfcc_scaled_red2)):
#     x.append(all_mfcc_scaled_red2[i][0])
# for i in range(len(all_mfcc_scaled_red2)):
#     y.append(all_mfcc_scaled_red2[i][1])
# for i in range(len(all_chroma_scaled_red1)):
#     z.append(all_mfcc_scaled_red2[i][0])
print(pca_chroma1)

feature1 = []
feature2 = []
for i in range(len(pca_mfcc2x)):
    feature1.append('mfcc')
for i in range(len(pca_mfcc1)):
    feature2.append('chroma')

df1 = pd.DataFrame({'x': np.asarray(pca_mfcc2x), 'y': np.asarray(pca_mfcc2y)/5, 'color': np.asarray(feature1)})
df2 = pd.DataFrame({'x': ((np.asarray(pca_mfcc1)/5)+0.6), 'y': np.asarray(pca_chroma1), 'color': np.asarray(feature2)})
print(df1)
print(df2)

viz1 = alt.Chart(df1).mark_point(opacity=0.6).encode(x='x', y='y', color='color:N').interactive()
#.configure_mark(opacity=0.5, color='cyan').interactive()
viz2 = alt.Chart(df2).mark_point(opacity=0.6).encode(x='x', y='y', color='color:N').interactive()
#.configure_mark(opacity=0.5, color='magenta').interactive()
viz1+viz2
#x=alt.X('x:Q', title='MFCC1'),
#y=alt.Y('y:Q', title='MFCC2'),
#z=alt.Z('z:Q', title='CHROMA')).add_selection(brush)
#alt.condition(brush, 'location:N', alt.value('grey')),
#     tooltip=['filename', 'location', 'ethnic_group', 'original_format', 'recorded_from_date', 'recording_context', ],
#     href='local_path'

#plt.scatter(np.asarray(pca_mfcc2x), np.asarray(pca_mfcc2y)/3, c='c', alpha=0.5)
#plt.scatter(np.asarray(pca_mfcc1)/8, np.asarray(pca_chroma1), c='m', alpha=0.5)
#plt.show()
