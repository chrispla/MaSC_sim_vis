import glob
import numpy as np
import librosa
import os
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
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

artists = []
titles = []
directories = []


#------FUNCTIONS------#			

def compute_MFCCs(y, sr):

	#MFCCs
	MFCCs = librosa.feature.mfcc(y = y, sr = sr, hop_length = 2205, n_mfcc = 13)

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


def plot(input_file, artists, titles, directories):

	#Values for each input
	x = []
	y = []

	file = open(input_file)
	for line in file.readlines():
		l = line.split(',')
		l.remove('\n')
		x.append(float(l[0]))
		y.append(float(l[1]))

	s1 = ColumnDataSource(data=dict(
	    x = x,
	    y = y,
	    desc = artists,
	    titles = titles,
	    directories = directories,
	))
	#Output file
	output_file(input_file+'.html')

	#Hover Tool
	hover = HoverTool(
		tooltips=[
			("Title", "@titles"),
			("Artist & Album", "@desc"),
		]
	)

	taptoolcallback = CustomJS(args=dict(source=s1),code = """

		var names = source.data['directories'];
		
	    var inds = source['selected']['1d'].indices;
	    var title = names[inds[0]];
	    title = title.slice(2,);

	    var para = document.createElement("p");
	    var node = document.createTextNode(title);
	    para.appendChild(node);
	    document.body.appendChild(para);

	    var x = document.createElement("AUDIO");
	    var song = String(title);
	    x.setAttribute("src",song);
	    x.setAttribute("controls", "controls");

	    document.body.appendChild(x);

	    var para2 = document.createElement("br");
	    document.body.appendChild(para2);
	    
	""")

	tap = TapTool(callback = taptoolcallback)


	p1 = figure(tools=[hover,"lasso_select", "reset", tap, "wheel_zoom"], title="Music Collections")
	p1.circle('x', 'y', source=s1, alpha = 0.7, size=5)

	s2 = ColumnDataSource(data=dict(artists=[], counts=[]))

	p2 = figure(x_range=(0,52),  y_range = artists, height = 10000, width = 1500, toolbar_location = None, title="Artists Info")
	p2.hbar(y='artists', right = 'counts', height = 0.4, source=s2)

	p2.xgrid.grid_line_color = None
	p2.legend.orientation = "horizontal"
	p2.legend.location = "top_center"

	s1.callback = CustomJS(args=dict(s2=s2), code="""

		var inds = cb_obj.selected['1d'].indices;
		var d1 = cb_obj.data;
		var d2 = s2.data;
		d2['artists'] = [];
		d2['counts'] = [];

		for (i=0; i<inds.length; i++){

			var current = d1['arts'][inds[i]];

			if (d2['artists'].indexOf(current) == -1){
				d2['artists'].push(d1['arts'][inds[i]]);
				d2['counts'].push(1);

			}
			else{
				d2['counts'][d2['artists'].indexOf(current)] += 1;
			}
			
		}

		s2.change.emit();


		""")

	p2.legend.location = "top_left"
#	layout = row(p1, p2)

#	show(layout)

	show(p1)

def cluster(input):

	clustered = KMeans().fit_transform(input)

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


duration = float(input('--- Enter duration of sound clip around its center:\n'))


#PROCESSES
answer_MFCCs = input('1. Do you want to compute the MFCCs? y/n\n')

answer_STFTs = input('2. Do you want to compute the power spectrogram? y/n\n')

answer_chroma = input('3. Do you want to compute the Constant-Q chromagram? y/n\n')

answer_plot_MFCCs = input('--- Do you want to plot the MFCCs and Chromagram?\n')

answer_plot_STFTs = input('--- Do you want to plot the STFTs and Chromagram?\n')




#---------MAIN---------#

if (answer_MFCCs in positive) or (answer_STFTs in positive) or (answer_chroma in positive):
	l = []
	counter = 1
	for root, dirs, files in os.walk('.'):
		for name in files:
			if '.wav' in name:
				filedir = os.path.join(root, name)
				
				#Used for plotting
				artists.append(filedir[:-len(name)-1])
				titles.append(name[:-4])
				directories.append(filedir)
			
				#File's duration
				total_duration = librosa.get_duration(filename = filedir) #load the whole file		
				offset = (total_duration/2) - duration/2 #starting point 
			
				#Load file	
				y, sr = librosa.load(filedir, offset = offset, duration = duration, sr = 44100 )	
				
				#Compute
				if answer_MFCCs in positive:
					compute_MFCCs(y, sr)
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


	#Join & plot MFCCs and chromagram
	if answer_plot_MFCCs in positive:

		print('Joining MFCCs and chromagram...')
		MFCCs_chroma = np.append(scaled_MFCCs[0], scaled_chroma[0])
		for row in range(n_files-1):
			temp_row = np.append(scaled_MFCCs[row+1], scaled_chroma[row+1])
			MFCCs_chroma = np.vstack((MFCCs_chroma, temp_row))

		#Dimensionality reduction to d=2
		MFCCs_chroma_reduced = TSNE(n_components=2).fit_transform(MFCCs_chroma)

		#Write the reduced array to a .csv
		file = open('MFCCs_reduced.csv', 'a', encoding='utf-8')
		for i in range(MFCCs_chroma_reduced.shape[0]):
			for j in range(MFCCs_chroma_reduced.shape[1]):
				file.write(str(MFCCs_chroma_reduced[i,j]))
				file.write(',')
			file.write('\n')
		file.close()

		#Plot
		print('Plotting MFFCs and chromagram...')
		plot('MFCCs_reduced.csv', artists, titles, directories)


	#Join & plot STFTs and chromagram
	if answer_plot_STFTs in positive:
		print('Joining STFTs and chromagram...')
		STFTs_chroma = np.append(scaled_STFTs[0], scaled_chroma[0])
		for row in range(n_files-1):
			temp_row = np.append(scaled_STFTs[row+1], scaled_chroma[row+1])
			STFTs_chroma = np.vstack((STFTs_chroma, temp_row))

		#Dimensionality reduction to d=2
		STFTs_chroma_reduced = TSNE(n_components=2).fit_transform(STFTs_chroma)

		#Write the reduced array to a .csv
		file = open('STFTs_reduced.csv', 'a', encoding='utf-8')
		for i in range(STFTs_chroma_reduced.shape[0]):
			for j in range(STFTs_chroma_reduced.shape[1]):
				file.write(str(STFTs_chroma_reduced[i,j]))
				file.write(',')
			file.write('\n')
		file.close()

		#Plot
		print('Plotting STFTs and chromagram...')
		plot('STFTs_reduced.csv', artists, titles, directories)


