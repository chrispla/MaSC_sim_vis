import glob
import numpy as np
import librosa
import os
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import CustomJS, ColumnDataSource, HoverTool, TapTool, LassoSelectTool

#-------HELPERS-------#

#Positive responses to questions
positive = ['y', 'Y', 'yes', 'Yes', 'YES']

#Find how many .wav files there are
n_files = 0
for root, dirs, files in os.walk('.'):
	for name in files:
		if '.wav' in name:
			n_files += 1



#------FUNCTIONS------#			

def compute_MFCCs(y, sr, n_mfcc):

	#MFCCs
	MFCCs = librosa.feature.mfcc(y = y, sr = sr, hop_length = 2205, n_mfcc = n_mfcc)

	#Write the ndarray to a .csv
	file = open('MFCCs.csv', 'a', encoding='utf-8')
	for i in range(MFCCs.shape[0]):
		for j in range(MFCCs.shape[1]):
			file.write(str(MFCCs[i,j]))
			file.write(',')
	file.write('\n')
	file.close()

def compute_STFTs(y):

	#STFTs
	STFTs = librosa.core.stft(y = y)

	#Convert the power spectrogram to decibel units
	STFTs_db = librosa.core.power_to_db(S = STFTs)

	#Write the ndarray to a .csv
	file = open('STFTs_db.csv', 'a', encoding='utf-8')
	for i in range(STFTs_db.shape[0]):
		for j in range(STFTs_db.shape[1]):
			file.write(str(STFTs_db[i,j]))
			file.write(',')
	file.write('\n')
	file.close()

def compute_chroma(y, sr):

	#chroma
	chroma = librosa.feature.chroma_cqt(y = y, sr = sr, hop_length = 512, n_chroma = 48, n_octaves = 7)

	#Write the ndarray to a .csv
	file = open('chroma.csv', 'a', encoding='utf-8')
	for i in range(chroma.shape[0]):
		for j in range(chroma.shape[1]):
			file.write(str(chroma[i,j]))
			file.write(',')
	file.write('\n')
	file.close()


def standardize(input_file, output_file):

	#Read file to numpy array
	data = []			
	file = open(input_file)	
	for line in file.readlines():
		l = line.split(',')
		l.remove('\n')
		data.append(l)
	file.close()
	data = np.array(data)

	#Standardize
	scaler = StandardScaler()
	scaled = scaler.fit_transform(data)

	file = open(output_file, 'a', encoding='utf-8')
	for i in range(scaled.shape[0]):
		for j in range(scaled.shape[1]):
			file.write(str(scaled[i,j]))
			file.write(',')
		file.write('\n')
	file.close()

	return scaled


def plot(input_file):

	#Values for each input
	x = []
	y = []

	file = open(input_file)
	for line in file.readlines():
		l = line.split(',')
		l.remove('\n')
		x.append(l[0])
		y.append(l[1])


	#Output file
	output_file("MusicCollections.html")

	#Hover Tool
	hover = HoverTool(
        tooltips=[
            ("(x,y)", "($x, $y)"),
            ("desc", "@desc"),
        ]
    )

	#Axes Range
	x_range = [float(min(x))-10, float(max(x))+10]
	y_range = [float(min(y))-10, float(max(y))+10]

	#Plot
	p = figure(
		tools = [hover, "lasso_select", "reset", "wheel_zoom"],
		title = "Music Collections",
		y_range = y_range,
		x_range = x_range)

	#Renderers
	p.circle(x, y, radius = 0.3, alpha = 0.5, fill_color = "black", size = 8)

	show(p)

#	s1 = ColumnDataSource(data = dict(x = x, y = y, desc = names, arts = artists))
#	p1 = figure(tools = [hover,"lasso_select", "reset", tap, "wheel_zoom"], title = "Music Collections")


#------PARAMETERS------#

#UNIVERSAL
#sr = int(input('Enter sample rate: [Type 0 for default (44100)]\n'))
#if sr == 0:
#	m_sr = 44100

#hop_length = int(input('Enter hop length: [Type 0 for default (2205)]\n'))
#if hop_length == 0:
#	hop_length = 2205


duration = float(input('Enter duration of sound clip around its center:\n'))


#PROCESSES
answer_MFCCs = input('Do you want to compute the MFCCs? y/n\n')
if answer_MFCCs in positive:
	#MFCC
	n_mfcc = int(input('---Enter number of MFCCs: [Type 0 for default (13)]\n'))
	if n_mfcc == 0:
		n_mfcc = 13

answer_STFTs = input('Do you want to compute the power spectrogram? y/n\n')

answer_chroma = input('Do you want to compute the Constant-Q chromagram? y/n\n')




#---------MAIN---------#


l = []
counter = 1
for root, dirs, files in os.walk('.'):
	for name in files:
		if '.wav' in name:
			filedir = os.path.join(root, name)
		
			#File's duration
			total_duration = librosa.get_duration(filename = filedir) #load the whole file		
			offset = (total_duration/2) - duration/2 #starting point 
		
			#Load file	
			y, sr = librosa.load(filedir, offset = offset, duration = duration, sr = 44100 )	
			
			#Compute
			if answer_MFCCs in positive:
				compute_MFCCs(y, sr, n_mfcc)
			if answer_STFTs in positive:
				compute_STFTs(y)
			if answer_chroma in positive:
				compute_chroma(y, sr)
			
			#Progress bar & function
			print('[', counter, '/', n_files, '] ', 'Computing for ', name, sep='')
			counter +=1


#Standardization
if answer_MFCCs in positive:
	print('Standardizing MFCCs...')
	scaled_MFCCs = standardize('MFCCs.csv', 'MFCCs_Standardized.csv')

if answer_STFTs in positive:
	print('Standardizing STFTs...')
	scaled_STFTs = standardize('STFTs_db.csv', 'STFTs_db_Standardized.csv')

if answer_chroma in positive:
	print('Standardizing chromagram...')
	scaled_chroma = standardize('chroma.csv', 'chroma_Standardized.csv')


#Join the 3 matrices to 1
if (answer_MFCCs in positive) and (answer_STFTs in positive) and (answer_chroma in positive):
	print('Joining MFCCs, STFTs, and chromagram...')
	joint = np.append(scaled_MFCCs[0], scaled_STFTs[0])
	joint = np.append(joint, scaled_chroma[0])
	for row in range(n_files-1):
		temp_row = np.append(scaled_MFCCs[row+1], scaled_STFTs[row+1])
		temp_row = np.append(temp_row, scaled_chroma[row+1])
		joint = np.vstack((joint, temp_row))


#Dimensionality reduction to d=2
print('Reducing dimesions to 2...')
joint_reduced = TSNE(n_components=2).fit_transform(joint)


#Write the array to a .csv
file = open('joint.csv', 'a', encoding='utf-8')
for i in range(joint.shape[0]):
	for j in range(joint.shape[1]):
		file.write(str(joint[i,j]))
		file.write(',')
	file.write('\n')
file.close()

#Write the reduced array to a .csv
file = open('joint_reduced.csv', 'a', encoding='utf-8')
for i in range(joint_reduced.shape[0]):
	for j in range(joint_reduced.shape[1]):
		file.write(str(joint_reduced[i,j]))
		file.write(',')
	file.write('\n')
file.close()

#Plot
print('Plotting...')
plot('joint_reduced.csv')

