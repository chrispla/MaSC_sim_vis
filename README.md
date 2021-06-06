# MasC Compendium Visualization
#### A collection of scripts for visualizing the Arab Mashriq collection of the NYU Abu Dhabi Library and the Eisenberg collection

```mfcc_t-SNE.ipynb```: Compute MFCC from audio, reduce dimension to 2 with t-SNE, and plot

```chromagram_t-SNE.ipynb```: Compute chromagram from audio, reduce dimension to 3 and 2 with t-SNE, and plot

```pca_t-SNE.ipynb```: Compute mel spectrogram from audio, do PCA using different number of components, reduce dimension to 2, and plot alongside intensity

```autoencoder_t_SNE```: Compute mel spectrogram from audio, take bottleneck of autoencoder, reduce dimension to 2, and plot alongside intensity

```compute_features.py```: Compute, save, and plot STFT, chroma, and MFCC

```startAD_demo.ipynb```: Independent similarity axis traversal visualization of MFCC and chroma for startAD demo

