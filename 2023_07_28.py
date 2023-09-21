#!/usr/bin/env python
# coding: utf-8

# 
# ## Yicheng May 2023

# In[1]:


from google.colab import drive
drive.mount('/content/drive')
get_ipython().run_line_magic('cd', '"/content/drive/MyDrive"')


# In[1]:


import h5py
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.fft import fft, fftfreq
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
from pylab import plot, xlabel, ylabel
from scipy import arange
from scipy import signal
from scipy import linalg
import time
import warnings
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
import pandas as pd
#import ipywidgets as widgets
#from ipywidgets import *
warnings.filterwarnings('ignore')


# Import libraries, define filenames and folder paths:

# In[2]:


def load_data(file_path,n):
    # Open the file
    f = h5py.File(file_path, 'r')
    # List all the attributes for the file
    attrs = list(f.attrs.keys())
    # List of the stations
    stations = list(f.keys())
    dset = f[stations[n]]
    # List of attributes for the station
    station_attr = list(dset.attrs.keys())
    length = dset.shape[1]/(100*60*60*24)
    # Access the distance in meters of the station from the event
    meters = [dset.attrs['dist_m'],
    dset.attrs['ele'],
    dset.attrs['lat'],
    dset.attrs['lon']]
    return attrs, stations, station_attr, meters, length

def distance_epicenter(file_path):
    f = h5py.File(file_path, 'r')
    distance_list={}
    for k in f.keys():
        distance_list[k] = f[k].attrs['dist_m']
    return distance_list

def extract_data(x,y,file_path):
    # Extract a portion of the data from numerous stations
    f = h5py.File(file_path, 'r')
    # List of stations to consider
    stations_list = list(f.keys())

    # Start and end index values for the portion of data to extract
    # x*3600*100 is x hours of data at 100 Hzone day of data.
    # this example outputs (y-x) hour of second day:
    start = x*3600*100
    end = y*3600*100

    # Number of channels
    num_channels = 3
    # Create 4 empty lists to store the extracted data
    data = [[] for _ in range(len(stations_list))]
    # Extract the data from the different stations
    for i in range(len(stations_list)):
        data[i] = f[stations_list[i]][0:num_channels,start:end]
    # Slice the data to every station at the first channel
    for i in range(len(stations_list)):
        data[i] = data[i][0]
    return data


# In[3]:


distance_epicenter("ev0001903830.h5")


# In[4]:


print(load_data("ev0001903830.h5",0))
stations_list = load_data("ev0001903830.h5",0)[1]


# In[5]:


data = extract_data(650,722,"ev0001903830.h5")


# In[6]:


data


# In[7]:


def taper(nt,t1,t2,alpha,cos,zeros,concat,x):
    wndo=[]; nts=t2-t1;r=alpha*float(nts);dat_w=[];
    for i in range(0,int(r/2)):
        ri=0.5*( 1+cos( 2*3.1416*(1/r)*(float(i)-r/2) ) )
        wndo.append(ri)
    for i in range(int(r/2)+1,nts-int(r/2)+1):
        wndo.append(1)
    for i in range(nts-int(r/2)+1,nts+1):
        ri=0.5*( 1+cos( 2*3.1416*(1/r)*(float(i)-nts+r/2) ) )
        wndo.append(ri)
    #wndo=tukey(t2-t1,alpha)
    padleft=zeros(t1);padright=zeros(nt-t2)
    wndo_a=concat((padleft,wndo,padright))
    for k in range(len(stations_list)):
        dat_ww=np.multiply(wndo_a, x[k])
        dat_w.append(dat_ww)
    return dat_w


# In[8]:


fs = 100
sampling_frequency = 100  # Number of samples per second
t = np.arange(0, 1, 1/fs)
fq = 100.0 # frequency of input signal
dt = 1/fs  # time sampling of input signal
f_min = 1  # desired cutoff for high-pass filter
nt = len(data[0])


# In[9]:


tt=dt*np.arange(nt);
To=nt*dt;
clip_w=0.1
# make taper and apply to signal:
dat_w=taper(nt,0,nt,clip_w,np.cos,np.zeros,np.concatenate,data)


# In[10]:


dat_w


# In[11]:


y=kkk


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
#plt.xlim([34500,35000])
for k in range(len(stations_list)):
    plt.plot(tt, dat_w[k], label='Tapered Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()


# In[12]:


def filter(order, cutoff_freq, sampling_freq,x):
    # Compute the digital Butterworth filter coefficients
    b, a = signal.butter(order, cutoff_freq / (sampling_freq / 2), btype='high', analog=False)

    # Apply the forward and backward filter to the input signal
    filtered_signal = signal.filtfilt(b, a, x)
    return filtered_signal

# Define the filter parameters
# order # Filter order
# cutoff_freq  # Cutoff frequency (in Hz)
# sampling_freq  # Sampling frequency (in Hz)
# x is the data

filtered_signal = filter(4, 1, 100, dat_w)


# In[ ]:


y=kkkk


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
for k in range(len(stations_list)):
    plt.plot(tt, filtered_signal[k], label='filtered Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    #plt.xlim([0,200])
    plt.legend()
    plt.grid(True)
    plt.show()


# 1. If the timeframe is 24 hours what should I use in X-axis standing for time
# 2. how do I deal with the extra large amplitude
# 3. how to plot the 24-hour signal
# 4. how deal with different stations
# 5. plot the label in barchart and plot the amplitude plot

# In[13]:


# Use 5 seconds time window as cutoff for the four stations and first channel
def datacutoff(x):
    datafiltered = list()
    for k in range(len(stations_list)):
        for i in range(int(len(x[k])/100/5)):#360000/100/5
            datafiltered.append(filtered_signal[k][500*i:500*(i+1)])
    return datafiltered


# In[14]:


datafiltered = datacutoff(data)
len(datafiltered)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
fig,ax=plt.subplots(3)
ax[0].plot(datafiltered[864],'k-')
ax[1].plot(datafiltered[1],'k-')
ax[2].plot(datafiltered[2],'k-')


# Use obspy library to apply 1HZ high-pass filter

# Featrue Extraction

# For each waveform, feature vectors are calculated to use in clustering analysis. The data features are time and frequency domain scalar values that include the integral of the squared waveform, maximum spectral amplitude, frequency at the maximum spectral amplitude, center frequency, signal bandwidth, zero upcrossing rate, and the rate of spectral peaks

# First using fourier transform to get the frequency domain values

# In[15]:


fs=100#HZ sampling freq
t= np.arange(0,5,1/fs) #time interval
f = 100;# signal freq???????????????????????????????????????
# generate frequency axis
n=np.size(t)# number of samples
fi = np.linspace(0,int(fs/2),int(n/2))#freq interval


 # Calculate the Fourier transform of the waveform
def myfourier(x):
    datafft = list()#spectral_amplitudes
    for i in range(len(x)):
        datafft.append(fft(x[i])[0: int(n/2)])# #spectral_amplitudes
    return datafft


# In[16]:


datafft = myfourier(datafiltered)
len(datafft)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
fig,ax=plt.subplots(2)
ax[0].plot(fi,np.abs(datafft[10])/len(datafft[10]));
ax[1].plot(datafiltered[10]);


# Get the intergral of squared waveform

# In[17]:


def iosw(x):
    integral = list()
    for i in range(len(x)):
        # Get the waveform data
        data0 = datafiltered[i]

        # Square the waveform data
        data_squared = data0 ** 2

        # Calculate the integral of the squared data
        integral.append(np.sum(data_squared))
    return integral
integral = iosw(datafiltered)


# Max spectral amplitude:

# Frequency of max spectral amplitude:

# In[18]:


#Assuming Fx is a list or array containing the spectral amplitudes and fi is a corresponding list or array containing the frequencies,
#this code finds the index of the maximum amplitude and retrieves the corresponding frequency.
def freq(x):
    max_indices = []
    max_amplitudes = []
    for array in x:
        max_index = np.argmax(array)
        max_indices.append(max_index)
        max_amplitudes.append(array[max_index].real)

    # Retrieve the corresponding frequencies
    frequencies = [fi[index] for index in max_indices]
    return frequencies, max_amplitudes
frequencies, max_amplitudes = freq(datafft)


# Center frequency:

# In[19]:


#Assuming fi and Fxi are lists or arrays representing frequencies and corresponding
#spectral amplitudes, respectively, this code calculates the center frequency
#using the given equation.
def cenfreq(x,y):
    center_frequency = list()
    for i in range(len(x)):
        numerator = np.sum(fi*y[i]).real
        denominator = np.sum(y[i]).real
        center_frequency.append(numerator / denominator)
    return frequencies
center_frequency = cenfreq(datafiltered,datafft)


# Signal bandwidth:

# In[20]:


# Assuming fi and Fxi are lists or arrays representing frequencies and corresponding spectral amplitudes,
# and fcenter is the center frequency, this code calculates the signal bandwidth using the given equation.
def signalb(x,y,z):
    signal_bandwidth = list()
    for i in range(len(x)):
        numerator = np.sum((fi - y[i])**2)
        denominator = np.sum(z[i])
        signal_bandwidth.append(np.sqrt(numerator / denominator).real)
    return signal_bandwidth
signal_bandwidth = signalb(datafiltered,center_frequency,datafft)


# Zero up-crossing rate:

# In[21]:


# Assuming fi and Fxi are lists or arrays representing frequencies and corresponding
# spectral amplitudes, this code calculates the zero up-crossing rate using the given equation.
def zur(x,y):
    zero_upcrossing_rate = list()
    for i in range(len(x)):
        omega = 2 * np.pi * fi
        numerator = np.sum(omega**2 * y[i]**2)
        denominator = np.sum(y[i]**2)
        zero_upcrossing_rate.append(np.sqrt(numerator / denominator).real)
    return zero_upcrossing_rate
zero_upcrossing_rate = zur(datafiltered,datafft)


# Rate of spectral peaks:

# In[22]:


len(datafft)


# In[23]:


#Assuming fi and Fxi are lists or arrays representing frequencies and corresponding
#spectral amplitudes, this code calculates the rate of spectral peaks using the given equation.
def rosp(x,y):
    rate_of_spectral_peaks = list()
    for i in range(len(x)):
        omega = 2 * np.pi * fi
        numerator = np.sum(omega**4 * y[i]**2)
        denominator = np.sum(omega**2 * y[i]**2)
        rate_of_spectral_peaks.append(np.sqrt(numerator / denominator).real)
    return rate_of_spectral_peaks
rate_of_spectral_peaks = rosp(datafiltered,datafft)


# PCA

# In[24]:


def creatdf():
    df = pd.DataFrame(list(zip(integral, max_amplitudes, frequencies, center_frequency, signal_bandwidth,zero_upcrossing_rate,rate_of_spectral_peaks)),
                  columns = ['integral of the squared waveform', 'maximum spectral amplitude', 'frequency at the maximum spectral amplitude',
                             'center frequency', 'signal bandwidth', 'zero upcrossing rate', 'rate of spectral peaks'])
    return df
df = creatdf()
df


# In[25]:


# Assuming DataFrame is named 'df'
X = df

# Splitting the dataset into training and test sets
X_train, X_test = train_test_split(X, test_size=0.1, random_state=66)


# In[26]:


# Reset the index of the test data
test_data = X_test.reset_index()

# Get the position of the test data
indy = test_data['index'].tolist()

indy = sorted(indy)

indy = np.hstack(indy)

# Display the position of the test data
print("Position of test data:", indy)


# In[27]:


scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)


# In[28]:


import seaborn as sns
plt.figure(figsize=(7,7))
sns.heatmap(df.corr(),color = "k", annot=True)


# In[29]:


pca = PCA(n_components=7)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

# Checking the explained variance ratio
prop_var = pca.explained_variance_ratio_
eigenvalues = pca.explained_variance_
PC_numbers = np.arange(pca.n_components_) + 1

plt.plot(PC_numbers,
         prop_var,
         'ro-')
plt.title('Figure 1: Scree Plot', fontsize=8)
plt.ylabel('Proportion of Variance', fontsize=8)
plt.show()
pca.explained_variance_ratio_


# Whitening
# We have used PCA to reduce the dimension of the data. There is a closely related preprocessing step called whitening (or, in some other literatures, sphering) which is needed for some algorithms. If we are training on images, the raw input is redundant, since adjacent pixel values are highly correlated. The goal of whitening is to make the input less redundant; more formally, our desiderata are that our learning algorithms sees a training input where (i) the features are less correlated with each other, and (ii) the features all have the same variance.
# Whitening combined with dimensionality reduction. If you want to have data that is whitened and which is lower dimensional than the original input, you can also optionally keep only the top k
#  components of xPCAwhite
# . When we combine PCA whitening with regularization (described later), the last few components of xPCAwhite
#  will be nearly zero anyway, and thus can safely be dropped.
#  http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/
# 
#  https://learndataa.com/2020/09/15/data-preprocessing-whitening-or-sphering-in-python/

# In[30]:


def data_Whitening(x):
    # Zero center data
    xc = x - np.mean(x, axis=0)
    print(xc.shape)
    xc = xc.T
    print('xc.shape:', xc.shape, '\n')

    # Calculate Covariance matrix
    # Note: 'rowvar=True' because each row is considered as a feature
    # Note: 'bias=True' to divide the sum of squared variances by 'n' instead of 'n-1'
    xcov = np.cov(xc, rowvar=True, bias=True)
    print

    # Calculate Eigenvalues and Eigenvectors
    w, v = linalg.eig(xcov) # .eigh()
    # Note: Use w.real.round(4) to (1) remove 'j' notation to real, (2) round to '4' significant digits
    print("Eigenvalues:\n", w.real.round(4), '\n')
    print("Eigenvectors:\n", v, '\n')

    # Calculate inverse square root of Eigenvalues
    # Optional: Add '.1e5' to avoid division errors if needed
    # Create a diagonal matrix
    diagw = np.diag(1/(w**0.5)) # or np.diag(1/((w+.1e-5)**0.5))
    diagw = diagw.real.round(4) #convert to real and round off
    print("Diagonal matrix for inverse square root of Eigenvalues:\n", diagw, '\n')

    # Calculate Rotation (optional)
    # Note: To see how data can be rotated
    xrot = np.dot(v, xc)

    # Whitening transform using PCA (Principal Component Analysis)
    wpca = np.dot(np.dot(diagw, v.T), xc)
    wpca = wpca[:-1].T
    return wpca
wpca = data_Whitening(X_train_pca)
wpca_test = data_Whitening(X_test_pca)
wpca


# how to separete the clusters/is there any noises in it  /found some precuser/ introduce the method relevant for earthquke forcast potential/ area in twon and noise is natural earthquake and sounds
# why chooose bca this seems work explore this method bca potential,nonetropic

# In[ ]:


y = kkkk


# KMean

# In[ ]:


# Elbow Method for optimal number of clusters
Sum_of_squared_distances = []
K = range(2,21)
for num_clusters in K :
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(wpca)
    Sum_of_squared_distances.append(kmeans.inertia_)
plt.plot(K,Sum_of_squared_distances,'bx-')
plt.xlabel('Values of K')
plt.ylabel('Sum of squared distances/Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[ ]:


# Silhouette Method for optimal number of clusters

silhouette_avg = []
for num_clusters in K:
  # initialise kmeans
  kmeans = KMeans(n_clusters=num_clusters)
  kmeans.fit(wpca)
  cluster_labels = kmeans.labels_
  # silhouette score
  silhouette_avg.append(silhouette_score(wpca, cluster_labels))
plt.plot(K,silhouette_avg,'bx-')
plt.xlabel('Values of K')
plt.ylabel('Silhouette score')
plt.title('Silhouette analysis For Optimal k')
plt.show()


# In[ ]:


get_ipython().system('pip install git+https://github.com/milesgranger/gap_statistic.git')


# Gap Statistic

# In[ ]:


from gap_statistic import OptimalK

def KMeans_clustering_func(X, k):
    """
    K Means Clustering function, which uses the K Means model from sklearn.

    These user-defined functions *must* take the X (input features) and a k
    when initializing OptimalK
    """

    # Include any clustering Algorithm that can return cluster centers

    m = KMeans(random_state=12, n_clusters=k, init="k-means++")
    m.fit(X)

    # Return the location of each cluster center,
    # and the labels for each point.
    return m.cluster_centers_, m.predict(X)
#--------------------create a wrapper around OptimalK to extract cluster centers and cluster labels
optimalK = OptimalK(clusterer=KMeans_clustering_func)
#--------------------Run optimal K on the input data (subset_scaled_interim) and number of clusters
n_clusters = optimalK(wpca, cluster_array=np.arange(2, 21))
#--------------------Gap Statistics data frame
optimalK.gap_df


# The gap statistic is the difference in the dispersion of the clusters from the data features and null features. Clusters are formed using 2–20 centroids, and the method is reinitialized for 100 iterations using different randomly chosen centroid seeds, with the final model having the lowest dispersion from the cluster centers. For each of the 2–20 number of centroids, 19 total, the process is repeated 500 times using a population of 15,000 ran- domly selected data features to assess the total inertia mean and deviation for each number of clusters.

# In[ ]:


def gaps():
    gap = []
    gap_avg = []
    for k in range(2, 21):
        for i in range(5):
            #--------------------create a wrapper around OptimalK to extract cluster centers and cluster labels
            optimalK = OptimalK(clusterer=KMeans_clustering_func)
            #--------------------Run optimal K on the input data (subset_scaled_interim) and number of clusters
            n_clusters = optimalK(wpca, cluster_array=np.arange(k, k+1))
            #--------------------Gap Statistics data frame
            optimalK.gap_df[['n_clusters', 'gap_value']]
            gap.append(optimalK.gap_df['gap_value'])
    chunk_size = 5
    for i in range(0, len(gap), chunk_size):
        mean_value = np.mean(gap[i:i+chunk_size])
        gap_avg.append(mean_value)
    # Calculate the differences between consecutive gap values
    gap_differences = np.diff(gap_avg)
    data_points = gap_avg
    rate_of_change = np.divide(gap_differences, gap_avg[:-1])
    return gap_avg, data_points, rate_of_change
gap_avg, data_points, rate_of_change = gaps()


# In[ ]:


# The gap statistic
# (solid blue line) showing the difference in centroid dispersion. The rate of change shown
# as the dashed black line.

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(range(len(data_points)), data_points, 'bo-', label='Data Points')
plt.xticks(range(1, 20))

ax2.plot(range(2,20), rate_of_change, 'k--', label='Rate of Change')

ax1.set_xlabel('Index')
ax1.set_ylabel('Gap')
ax2.set_ylabel('$\Delta Gap$')
plt.title('Data Points and Rate of Change')
ax1.legend()
ax2.legend()

ax2.tick_params(axis='y', labelcolor='k')


# In[ ]:


wpca_test.shape


# In[31]:


# Set the number of clusters
num_clusters = 5

# Set the number of data points in each batch for centroid initialization
batch_size = 15000

# Set the maximum number of iterations without improvement
max_iterations = 100
NUM_ATTEMPTS = 500 # we will attempt this with 500 random initial centroids (NUM_ATTEMPTS, or m)

def bcentriods(data,num_clusters,batch_size,max_iterations,NUM_ATTEMPTS):
    final_cents = []
    final_inert = []

    for sample in range(NUM_ATTEMPTS):
        #print('\nCentroid attempt: ', sample) # Select batches of 50 randomly chosen data points
        random_indices = np.random.choice(len(data), size=batch_size)
        data_sample = data[random_indices]
        km = KMeans(n_clusters=num_clusters, init='random', max_iter=1, n_init=1)#, verbose=1)
        km.fit(data_sample)
        inertia_start = km.inertia_
        intertia_end = 0
        cents = km.cluster_centers_

        for iter in range(max_iterations):
            km = KMeans(n_clusters=num_clusters, init=cents, max_iter=1, n_init=1)
            km.fit(data_sample)
            #print('Iteration: ', iter)
            #print('Inertia:', km.inertia_)
            #print('Centroids:', km.cluster_centers_)
            inertia_end = km.inertia_
            cents = km.cluster_centers_
        final_cents.append(cents)
        final_inert.append(inertia_end)
        #print('Difference between initial and final inertia: ', inertia_start-inertia_end)
        # Get best centroids to use for full clustering
        best_cents = final_cents[final_inert.index(min(final_inert))]
    return best_cents
best_cents = bcentriods(wpca,5,15000,100,500)


# In[32]:


# Get best centroids to use for full clustering

km_full = KMeans(n_clusters=num_clusters, init=best_cents, max_iter=100, verbose=1, n_init=1)
km_full.fit(wpca_test)
# Assuming you have your evaluation data stored in 'evaluation_data' variable
evaluation_data = wpca_test
# Get the labels assigned to each data point using the best model
labels = km_full.predict(evaluation_data)

# Count the number of data points in each cluster
cluster_counts = [0] * num_clusters
for label in labels:
    cluster_counts[label] += 1

# Calculate the percentage of data points in each cluster
total_points = len(evaluation_data)
cluster_percentages = [(count / total_points) * 100 for count in cluster_counts]

# Print the cluster labels and their percentages
for i in range(num_clusters):
    print("Label {}: {:.2f}%".format(i+1, cluster_percentages[i]))


# https://towardsdatascience.com/iterative-initial-centroid-search-via-sampling-for-k-means-clustering-2b505119ae37
# 
# https://www.askpython.com/python/examples/plot-k-means-clusters-python

# In[33]:


def labelcount(x):
    count = []
    # Get the unique labels
    unique_labels = np.unique(x)
    # Iterate over the unique labels and access the data points for each label
    for label in unique_labels:
        data_points = np.where(labels == label)[0]
        print(f"Label {label}: {data_points}")
        count.append(len(data_points))
    return count, unique_labels
count,unique_labels = labelcount(labels)


# In[34]:


labels=labels.astype(int)
labels


# In[ ]:


count


# In[35]:


# Iterate over the unique labels and access the data points for each label
for label in unique_labels:
    data_points = evaluation_data[labels == label]
    print(f"Label {label}:")
    print(data_points)


# In[36]:


nn=len(data)
nn


# In[37]:


def time_interval(x):
    # Initialize an empty list for indy
    interval = []

    # Get the unique labels
    unique_labels = np.unique(x)

    # Iterate over the unique labels and access the data points for each label
    for label in unique_labels:
        # Find the indices where the label occurs in the 'labels' array
        indices = np.where(x == label)[0]
        # Append the indices corresponding to each label to the 'indy' list
        interval.append(indices)

    # Concatenate the elements in the 'indy' list to create the final 'indy' array
    indy = np.hstack(interval)*500
    return indy, interval
indy, interval = time_interval(labels)




interval = np.hstack(interval)
indy0=interval[:count[0]]
indy1=interval[(count[0]):(count[0]+count[1])]
indy2=interval[(count[0]+count[1]):(count[0]+count[1]+count[2])]
indy3=interval[(count[0]+count[1]+count[2]):(count[0]+count[1]+count[2]+count[3])]
indy4=interval[(count[0]+count[1]+count[2]+count[3]):(count[0]+count[1]+count[2]+count[3]+count[4])]

indy.shape


# In[ ]:


efft=[]
evaluation_data[indy0]
for i in range(len(evaluation_data[indy0])):
    efft.append(fft(evaluation_data[indy0][i])[0: int(n/2)].real.tolist())
len(efft)


# In[38]:


int_lab=labels.tolist()
int_lab0=int_lab[:count[0]]
indy


# In[ ]:


y=kkkk


# The time interval should be a bigger time period like one day or one hour(10%) with 10 days or 10 hours data (90%)

# In[ ]:



cols = ['royalblue','lightgreen','yellow','crimson','darkorange']
get_ipython().run_line_magic('matplotlib', 'notebook')
plt.figure(figsize=(7,7))
for v in range(len(stations_list)):
    for i in range(len(indy)-1):
        plot(np.arange(indy[i],indy[i+1]),data[v][indy[i]:indy[i+1]],cols[int_lab[i]])



# We have like 10% data labels how do we applied to all the data?
# 2. indy stands for time interval how many should we have in my data I just label different data points and how do I know which time interval it belong to,
# the key is how to define intervals (indy)

# 1. The 1 min interval starting at 18:09 shows all labels with a microseismic event occurring at 42 s with the P wave as Label 4 and the S wave as Label 2 (Figure 4c). Emergent waveforms are identiﬁed and assigned different labels (Figure 4d), implying that the feature vectors to describe each wavelet to data are adequately describing the key spectral properties of the waveforms.
# 
# 2. what does this model do and its goal

# In[ ]:


y=kkkkkk


# In[39]:


# 1 Choose the middle of the month to test the cluster model
finaldata = extract_data(749,750,"ev0001903830.h5")

# 2 make taper and apply to signal:
nt = len(finaldata[0])
dat_w=taper(nt,0,nt,clip_w,np.cos,np.zeros,np.concatenate,finaldata)


# 3 high pass filter
filtered_signal = filter(4, 1, 100, dat_w)

# 4.Use 5 seconds time window as cutoff for each station and first channel

datafiltered = datacutoff(finaldata)

indylist = []
int_lablist = []

for j in range(len(stations_list)):
    # divide the data into each station
    chunk_start = 720 * j
    chunk_end = 720 * (j + 1)
    h = datafiltered[chunk_start:chunk_end]

    # 5 Calculate the Fourier transform of the waveform
    datafft = myfourier(h)

    # 6 Get the intergral of squared waveform
    integral = iosw(h)

    # 7 Get Max spectral amplitude & Frequency of max spectral amplitude

    frequencies, max_amplitudes = freq(datafft)

    # 8 Get Center frequency:

    center_frequency = cenfreq(h,datafft)

    # 9 Signal bandwidth:

    signal_bandwidth = signalb(h,center_frequency,datafft)

    # 10 Zero up-crossing rate:
    zero_upcrossing_rate = zur(h,datafft)

    # 11 Rate of spectral peaks:
    rate_of_spectral_peaks = rosp(h,datafft)

    df = creatdf()

    # 12 PCA
    scaler = StandardScaler()
    finalX = scaler.fit_transform(df)
    pca = PCA(n_components=7)
    finalX_pca = pca.fit_transform(finalX)

    # 13 Whitening

    finalwpca = data_Whitening(finalX_pca)

    # 14 K-means

    km_full.fit(finalwpca)

    # Get the labels assigned to each data point using the best model
    labels = km_full.predict(finalwpca)

    # Count the number of data points in each cluster
    cluster_counts = [0] * num_clusters
    for label in labels:
        cluster_counts[label] += 1

    # Calculate the percentage of data points in each cluster
    total_points = len(finalwpca)
    cluster_percentages = [(count / total_points) * 100 for count in cluster_counts]

    # Print the cluster labels and their percentages
    for i in range(num_clusters):
        print("Label {}: {:.2f}%".format(i+1, cluster_percentages[i]))
    labels = labels.astype(int)

    # 15 Get the labels

    count,unique_labels = labelcount(labels)
    int_lab = labels.tolist()
    indy, interval = time_interval(labels)
    indylist.append(indy)
    int_lablist.append(int_lab)

# 16 Plot the results for each station

cols = ['royalblue','lightgreen','gold','crimson','darkorange']
get_ipython().run_line_magic('matplotlib', 'inline')
fig,ax=plt.subplots(len(stations_list),figsize=(30, 25))
for v in range(len(stations_list)):
    for i in range(len(indylist[v])-1):
        ax[v].plot(np.arange(indylist[v][i],indylist[v][i+1]),finaldata[v][indylist[v][i]:indylist[v][i+1]],cols[int_lablist[v][i]])


# In[46]:


# 1 Choose the middle of the month to test the cluster model
finaldata = extract_data(738,750,"ev0001903830.h5")

# 2 make taper and apply to signal:
nt = len(finaldata[0])
dat_w=taper(nt,0,nt,clip_w,np.cos,np.zeros,np.concatenate,finaldata)


# 3 high pass filter
filtered_signal = filter(4, 1, 100, dat_w)

# 4.Use 5 seconds time window as cutoff for each station and first channel

datafiltered = datacutoff(finaldata)

indylist = []
int_lablist = []

for j in range(len(stations_list)):
    # divide the data into each station
    chunk_start = 720 * j
    chunk_end = 720 * (j + 1)
    h = datafiltered[chunk_start:chunk_end]

    # 5 Calculate the Fourier transform of the waveform
    datafft = myfourier(h)

    # 6 Get the intergral of squared waveform
    integral = iosw(h)

    # 7 Get Max spectral amplitude & Frequency of max spectral amplitude

    frequencies, max_amplitudes = freq(datafft)

    # 8 Get Center frequency:

    center_frequency = cenfreq(h,datafft)

    # 9 Signal bandwidth:

    signal_bandwidth = signalb(h,center_frequency,datafft)

    # 10 Zero up-crossing rate:
    zero_upcrossing_rate = zur(h,datafft)

    # 11 Rate of spectral peaks:
    rate_of_spectral_peaks = rosp(h,datafft)

    df = creatdf()

    # 12 PCA
    scaler = StandardScaler()
    finalX = scaler.fit_transform(df)
    pca = PCA(n_components=7)
    finalX_pca = pca.fit_transform(finalX)

    # 13 Whitening

    finalwpca = data_Whitening(finalX_pca)

    # 14 K-means

    km_full.fit(finalwpca)

    # Get the labels assigned to each data point using the best model
    labels = km_full.predict(finalwpca)

    # Count the number of data points in each cluster
    cluster_counts = [0] * num_clusters
    for label in labels:
        cluster_counts[label] += 1

    # Calculate the percentage of data points in each cluster
    total_points = len(finalwpca)
    cluster_percentages = [(count / total_points) * 100 for count in cluster_counts]

    # Print the cluster labels and their percentages
    for i in range(num_clusters):
        print("Label {}: {:.2f}%".format(i+1, cluster_percentages[i]))
    labels = labels.astype(int)


# In[47]:


# 1 Choose the middle of the month to test the cluster model
finaldata = extract_data(24,36,"ev0001903830.h5")

# 2 make taper and apply to signal:
nt = len(finaldata[0])
dat_w=taper(nt,0,nt,clip_w,np.cos,np.zeros,np.concatenate,finaldata)


# 3 high pass filter
filtered_signal = filter(4, 1, 100, dat_w)

# 4.Use 5 seconds time window as cutoff for each station and first channel

datafiltered = datacutoff(finaldata)

indylist = []
int_lablist = []

for j in range(len(stations_list)):
    # divide the data into each station
    chunk_start = 720 * j
    chunk_end = 720 * (j + 1)
    h = datafiltered[chunk_start:chunk_end]

    # 5 Calculate the Fourier transform of the waveform
    datafft = myfourier(h)

    # 6 Get the intergral of squared waveform
    integral = iosw(h)

    # 7 Get Max spectral amplitude & Frequency of max spectral amplitude

    frequencies, max_amplitudes = freq(datafft)

    # 8 Get Center frequency:

    center_frequency = cenfreq(h,datafft)

    # 9 Signal bandwidth:

    signal_bandwidth = signalb(h,center_frequency,datafft)

    # 10 Zero up-crossing rate:
    zero_upcrossing_rate = zur(h,datafft)

    # 11 Rate of spectral peaks:
    rate_of_spectral_peaks = rosp(h,datafft)

    df = creatdf()

    # 12 PCA
    scaler = StandardScaler()
    finalX = scaler.fit_transform(df)
    pca = PCA(n_components=7)
    finalX_pca = pca.fit_transform(finalX)

    # 13 Whitening

    finalwpca = data_Whitening(finalX_pca)

    # 14 K-means

    km_full.fit(finalwpca)

    # Get the labels assigned to each data point using the best model
    labels = km_full.predict(finalwpca)

    # Count the number of data points in each cluster
    cluster_counts = [0] * num_clusters
    for label in labels:
        cluster_counts[label] += 1

    # Calculate the percentage of data points in each cluster
    total_points = len(finalwpca)
    cluster_percentages = [(count / total_points) * 100 for count in cluster_counts]

    # Print the cluster labels and their percentages
    for i in range(num_clusters):
        print("Label {}: {:.2f}%".format(i+1, cluster_percentages[i]))
    labels = labels.astype(int)


# In[48]:


# 1 Choose the middle of the month to test the cluster model
finaldata = extract_data(36,39,"ev0001903830.h5")

# 2 make taper and apply to signal:
nt = len(finaldata[0])
dat_w=taper(nt,0,nt,clip_w,np.cos,np.zeros,np.concatenate,finaldata)


# 3 high pass filter
filtered_signal = filter(4, 1, 100, dat_w)

# 4.Use 5 seconds time window as cutoff for each station and first channel

datafiltered = datacutoff(finaldata)

indylist = []
int_lablist = []

for j in range(len(stations_list)):
    # divide the data into each station
    chunk_start = 720 * j
    chunk_end = 720 * (j + 1)
    h = datafiltered[chunk_start:chunk_end]

    # 5 Calculate the Fourier transform of the waveform
    datafft = myfourier(h)

    # 6 Get the intergral of squared waveform
    integral = iosw(h)

    # 7 Get Max spectral amplitude & Frequency of max spectral amplitude

    frequencies, max_amplitudes = freq(datafft)

    # 8 Get Center frequency:

    center_frequency = cenfreq(h,datafft)

    # 9 Signal bandwidth:

    signal_bandwidth = signalb(h,center_frequency,datafft)

    # 10 Zero up-crossing rate:
    zero_upcrossing_rate = zur(h,datafft)

    # 11 Rate of spectral peaks:
    rate_of_spectral_peaks = rosp(h,datafft)

    df = creatdf()

    # 12 PCA
    scaler = StandardScaler()
    finalX = scaler.fit_transform(df)
    pca = PCA(n_components=7)
    finalX_pca = pca.fit_transform(finalX)

    # 13 Whitening

    finalwpca = data_Whitening(finalX_pca)

    # 14 K-means

    km_full.fit(finalwpca)

    # Get the labels assigned to each data point using the best model
    labels = km_full.predict(finalwpca)

    # Count the number of data points in each cluster
    cluster_counts = [0] * num_clusters
    for label in labels:
        cluster_counts[label] += 1

    # Calculate the percentage of data points in each cluster
    total_points = len(finalwpca)
    cluster_percentages = [(count / total_points) * 100 for count in cluster_counts]

    # Print the cluster labels and their percentages
    for i in range(num_clusters):
        print("Label {}: {:.2f}%".format(i+1, cluster_percentages[i]))
    labels = labels.astype(int)


# In[49]:


# 1 Choose the middle of the month to test the cluster model
finaldata = extract_data(36,37,"ev0001903830.h5")

# 2 make taper and apply to signal:
nt = len(finaldata[0])
dat_w=taper(nt,0,nt,clip_w,np.cos,np.zeros,np.concatenate,finaldata)


# 3 high pass filter
filtered_signal = filter(4, 1, 100, dat_w)

# 4.Use 5 seconds time window as cutoff for each station and first channel

datafiltered = datacutoff(finaldata)

indylist = []
int_lablist = []

for j in range(len(stations_list)):
    # divide the data into each station
    chunk_start = 720 * j
    chunk_end = 720 * (j + 1)
    h = datafiltered[chunk_start:chunk_end]

    # 5 Calculate the Fourier transform of the waveform
    datafft = myfourier(h)

    # 6 Get the intergral of squared waveform
    integral = iosw(h)

    # 7 Get Max spectral amplitude & Frequency of max spectral amplitude

    frequencies, max_amplitudes = freq(datafft)

    # 8 Get Center frequency:

    center_frequency = cenfreq(h,datafft)

    # 9 Signal bandwidth:

    signal_bandwidth = signalb(h,center_frequency,datafft)

    # 10 Zero up-crossing rate:
    zero_upcrossing_rate = zur(h,datafft)

    # 11 Rate of spectral peaks:
    rate_of_spectral_peaks = rosp(h,datafft)

    df = creatdf()

    # 12 PCA
    scaler = StandardScaler()
    finalX = scaler.fit_transform(df)
    pca = PCA(n_components=7)
    finalX_pca = pca.fit_transform(finalX)

    # 13 Whitening

    finalwpca = data_Whitening(finalX_pca)

    # 14 K-means

    km_full.fit(finalwpca)

    # Get the labels assigned to each data point using the best model
    labels = km_full.predict(finalwpca)

    # Count the number of data points in each cluster
    cluster_counts = [0] * num_clusters
    for label in labels:
        cluster_counts[label] += 1

    # Calculate the percentage of data points in each cluster
    total_points = len(finalwpca)
    cluster_percentages = [(count / total_points) * 100 for count in cluster_counts]

    # Print the cluster labels and their percentages
    for i in range(num_clusters):
        print("Label {}: {:.2f}%".format(i+1, cluster_percentages[i]))
    labels = labels.astype(int)


# In[50]:


# 1 Choose the middle of the month to test the cluster model
finaldata = extract_data(726,750,"ev0001903830.h5")

# 2 make taper and apply to signal:
nt = len(finaldata[0])
dat_w=taper(nt,0,nt,clip_w,np.cos,np.zeros,np.concatenate,finaldata)


# 3 high pass filter
filtered_signal = filter(4, 1, 100, dat_w)

# 4.Use 5 seconds time window as cutoff for each station and first channel

datafiltered = datacutoff(finaldata)

indylist = []
int_lablist = []

for j in range(len(stations_list)):
    # divide the data into each station
    chunk_start = 720 * j
    chunk_end = 720 * (j + 1)
    h = datafiltered[chunk_start:chunk_end]

    # 5 Calculate the Fourier transform of the waveform
    datafft = myfourier(h)

    # 6 Get the intergral of squared waveform
    integral = iosw(h)

    # 7 Get Max spectral amplitude & Frequency of max spectral amplitude

    frequencies, max_amplitudes = freq(datafft)

    # 8 Get Center frequency:

    center_frequency = cenfreq(h,datafft)

    # 9 Signal bandwidth:

    signal_bandwidth = signalb(h,center_frequency,datafft)

    # 10 Zero up-crossing rate:
    zero_upcrossing_rate = zur(h,datafft)

    # 11 Rate of spectral peaks:
    rate_of_spectral_peaks = rosp(h,datafft)

    df = creatdf()

    # 12 PCA
    scaler = StandardScaler()
    finalX = scaler.fit_transform(df)
    pca = PCA(n_components=7)
    finalX_pca = pca.fit_transform(finalX)

    # 13 Whitening

    finalwpca = data_Whitening(finalX_pca)

    # 14 K-means

    km_full.fit(finalwpca)

    # Get the labels assigned to each data point using the best model
    labels = km_full.predict(finalwpca)

    # Count the number of data points in each cluster
    cluster_counts = [0] * num_clusters
    for label in labels:
        cluster_counts[label] += 1

    # Calculate the percentage of data points in each cluster
    total_points = len(finalwpca)
    cluster_percentages = [(count / total_points) * 100 for count in cluster_counts]

    # Print the cluster labels and their percentages
    for i in range(num_clusters):
        print("Label {}: {:.2f}%".format(i+1, cluster_percentages[i]))
    labels = labels.astype(int)


# In[ ]:


# 1 Choose the middle of the month to test the cluster model
finaldata = extract_data(12,36,"ev0001903830.h5")

# 2 make taper and apply to signal:
nt = len(finaldata[0])
dat_w=taper(nt,0,nt,clip_w,np.cos,np.zeros,np.concatenate,finaldata)


# 3 high pass filter
filtered_signal = filter(4, 1, 100, dat_w)

# 4.Use 5 seconds time window as cutoff for each station and first channel

datafiltered = datacutoff(finaldata)

indylist = []
int_lablist = []

for j in range(len(stations_list)):
    # divide the data into each station
    chunk_start = 720 * j
    chunk_end = 720 * (j + 1)
    h = datafiltered[chunk_start:chunk_end]

    # 5 Calculate the Fourier transform of the waveform
    datafft = myfourier(h)

    # 6 Get the intergral of squared waveform
    integral = iosw(h)

    # 7 Get Max spectral amplitude & Frequency of max spectral amplitude

    frequencies, max_amplitudes = freq(datafft)

    # 8 Get Center frequency:

    center_frequency = cenfreq(h,datafft)

    # 9 Signal bandwidth:

    signal_bandwidth = signalb(h,center_frequency,datafft)

    # 10 Zero up-crossing rate:
    zero_upcrossing_rate = zur(h,datafft)

    # 11 Rate of spectral peaks:
    rate_of_spectral_peaks = rosp(h,datafft)

    df = creatdf()

    # 12 PCA
    scaler = StandardScaler()
    finalX = scaler.fit_transform(df)
    pca = PCA(n_components=7)
    finalX_pca = pca.fit_transform(finalX)

    # 13 Whitening

    finalwpca = data_Whitening(finalX_pca)

    # 14 K-means

    km_full.fit(finalwpca)

    # Get the labels assigned to each data point using the best model
    labels = km_full.predict(finalwpca)

    # Count the number of data points in each cluster
    cluster_counts = [0] * num_clusters
    for label in labels:
        cluster_counts[label] += 1

    # Calculate the percentage of data points in each cluster
    total_points = len(finalwpca)
    cluster_percentages = [(count / total_points) * 100 for count in cluster_counts]

    # Print the cluster labels and their percentages
    for i in range(num_clusters):
        print("Label {}: {:.2f}%".format(i+1, cluster_percentages[i]))
    labels = labels.astype(int)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
cols = ['royalblue','lightgreen','gold','crimson','darkorange']

fig,ax=plt.subplots(len(stations_list),figsize=(30, 25))
for v in range(len(stations_list)):
    for i in range(len(indylist[v])-1):
        ax[v].plot(np.arange(indylist[v][i],indylist[v][i+1]),finaldata[v][indylist[v][i]:indylist[v][i+1]],cols[int_lablist[v][i]])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
cols = ['royalblue','lightgreen','gold','crimson','darkorange']
fig,ax=plt.subplots(len(stations_list),figsize=(30, 25))
for v in range(len(stations_list)):
    for i in range(len(indylist[v])-1):
        ax[v].plot(np.arange(indylist[v][i],indylist[v][i+1]),finaldata[v][indylist[v][i]:indylist[v][i+1]],cols[int_lablist[v][i]])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
cols = ['royalblue','lightgreen','gold','crimson','darkorange']


# In[ ]:


finaldata = extract_data(749,750,"ev0001903830.h5")


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
for i in range(len(indylist[16])-1):
    plot(np.arange(indylist[16][i],indylist[16][i+1]),finaldata[16][indylist[16][i]:indylist[16][i+1]],cols[int_lablist[16][i]])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
for i in range(len(indylist[22])-1):
    plot(np.arange(indylist[22][i],indylist[22][i+1]),finaldata[22][indylist[22][i]:indylist[22][i+1]],cols[int_lablist[22][i]])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
for i in range(len(indylist[49])-1):
    plot(np.arange(indylist[49][i],indylist[49][i+1]),finaldata[49][indylist[49][i]:indylist[49][i+1]],cols[int_lablist[49][i]])


# In[40]:


get_ipython().run_line_magic('matplotlib', 'notebook')
for i in range(len(indylist[43])-1):
    plot(np.arange(indylist[43][i],indylist[43][i+1]),finaldata[43][indylist[43][i]:indylist[43][i+1]],cols[int_lablist[43][i]])


# In[42]:


get_ipython().run_line_magic('matplotlib', 'notebook')
for i in range(len(indylist[11])-1):
    plot(np.arange(indylist[11][i],indylist[11][i+1]),finaldata[11][indylist[11][i]:indylist[11][i+1]],cols[int_lablist[11][i]])


# In[43]:



for i in range(len(indylist[23])-1):
    plot(np.arange(indylist[23][i],indylist[23][i+1]),finaldata[23][indylist[23][i]:indylist[23][i+1]],cols[int_lablist[23][i]])


# In[44]:


get_ipython().run_line_magic('matplotlib', 'notebook')
for i in range(len(indylist[15])-1):
    plot(np.arange(indylist[15][i],indylist[15][i+1]),finaldata[15][indylist[15][i]:indylist[15][i+1]],cols[int_lablist[15][i]])


# In[45]:


get_ipython().run_line_magic('matplotlib', 'notebook')
for i in range(len(indylist[3])-1):
    plot(np.arange(indylist[3][i],indylist[3][i+1]),finaldata[3][indylist[3][i]:indylist[3][i+1]],cols[int_lablist[3][i]])

