import mne
import numpy as np
import pywt
from mne.decoding import CSP
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score
import pandas as pd

# Mention the file path to the dataset
filename = "A01T.gdf"

# Read raw data from the file
raw = mne.io.read_raw_gdf(filename)

# Print information about the raw data
print(raw.info)

# Print channel names
print(raw.ch_names)

# Find the events time positions
events, _ = mne.events_from_annotations(raw)

# Pre-load the data
raw.load_data()

# Filter the raw signal with a band pass filter in 7-35 Hz
raw.filter(7., 35., fir_design='firwin')

# Remove the EOG channels and pick only desired EEG channels
raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False,
                       exclude='bads')

# Extract epochs of 3s time period from the dataset into 288 events for all 4 classes
tmin, tmax = 1., 4.
event_id = dict({'769': 7,'770': 8,'771': 9,'772': 10})
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True)

# Plot average of left hand epoch
evoked = epochs['769'].average()
print(evoked)
evoked.plot(time_unit='s')

# Plot average of right hand epoch
evoked = epochs['770'].average()
print(evoked)
evoked.plot(time_unit='s')

# Plot average of foot epoch
evoked = epochs['771'].average()
print(evoked)
evoked.plot(time_unit='s')

# Plot average of tongue epoch
evoked = epochs['772'].average()
print(evoked)
evoked.plot(time_unit='s')

# Getting labels and changing labels from 7,8,9,10 to 1,2,3,4
labels = epochs.events[:,-1] - 7 + 1 

data = epochs.get_data()

# Signal is decomposed to level 5 with 'db4' wavelet
def wpd(X): 
    coeffs = pywt.WaveletPacket(X,'db4',mode='symmetric',maxlevel=5)
    return coeffs

def feature_bands(x):
    Bands = np.empty((8,x.shape[0],x.shape[1],30)) # 8 frequency band coefficients are chosen from the range 4-32Hz
    for i in range(x.shape[0]):
        for ii in range(x.shape[1]):
             pos = []
             C = wpd(x[i,ii,:]) 
             pos = np.append(pos,[node.path for node in C.get_level(5, 'natural')])
             for b in range(1,9):
                 Bands[b-1,i,ii,:] = C[pos[b]].data
    return Bands

wpd_data = feature_bands(data)

# One-hot encoding labels
enc = OneHotEncoder()
X_out = enc.fit_transform(labels.reshape(-1,1)).toarray()

# Cross Validation Split
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

acc = []
ka = []
prec = []
recall = []

def build_classifier(num_layers=1):
    classifier = Sequential()
    # First Layer
    classifier.add(Dense(units=124, kernel_initializer='uniform', activation='relu', input_dim=32, 
                         kernel_regularizer=regularizers.l2(0.01))) # L2 regularization
    classifier.add(Dropout(0.5))
    # Intermediate Layers
    for itr in range(num_layers):
        classifier.add(Dense(units=124, kernel_initializer='uniform', activation='relu', 
                             kernel_regularizer=regularizers.l2(0.01))) # L2 regularization
        classifier.add(Dropout(0.5))   
    # Last Layer
    classifier.add(Dense(units=4, kernel_initializer='uniform', activation='softmax'))
    classifier.compile(optimizer='rmsprop' , loss='categorical_crossentropy', metrics=['accuracy'])
    return classifier

# Perform cross-validation
for train_idx, test_idx in cv.split(labels):
    Csp = []
    ss = []
    nn = []
    
    label_train, label_test = labels[train_idx], labels[test_idx]
    y_train, y_test = X_out[train_idx], X_out[test_idx]
    
    # Apply CSP filter separately for all frequency band coefficients
    Csp = [CSP(n_components=4, reg=None, log=True, norm_trace=False) for _ in range(8)]
    ss = preprocessing.StandardScaler()

    X_train = ss.fit_transform(np.concatenate(tuple(Csp[x].fit_transform(wpd_data[x,train_idx,:,:],label_train) for x in range(8)),axis=-1))
    X_test = ss.transform(np.concatenate(tuple(Csp[x].transform(wpd_data[x,test_idx,:,:]) for x in range(8)),axis=-1))
    
    nn = build_classifier()  
    nn.fit(X_train, y_train, batch_size=32, epochs=300)
    
    y_pred = nn.predict(X_test)
    pred = (y_pred == y_pred.max(axis=1)[:,None]).astype(int)

    acc.append(accuracy_score(y_test.argmax(axis=1), pred.argmax(axis=1)))
    ka.append(cohen_kappa_score(y_test.argmax(axis=1), pred.argmax(axis=1)))
    prec.append(precision_score(y_test.argmax(axis=1), pred.argmax(axis=1), average='weighted'))
    recall.append(recall_score(y_test.argmax(axis=1), pred.argmax(axis=1), average='weighted'))

# Create a DataFrame to store the evaluation scores
scores = {'Accuracy': acc, 'Kappa': ka, 'Precision': prec, 'Recall': recall}
Es = pd.DataFrame(scores)

# Compute average scores
avg = {'Accuracy': [np.mean(acc)], 'Kappa': [np.mean(ka)], 'Precision': [np.mean(prec)], 'Recall': [np.mean(recall)]}
Avg = pd.DataFrame(avg)

# Concatenate the evaluation scores and average scores DataFrames
T = pd.concat([Es, Avg])

# Set index labels
T.index = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'Avg']
T.index.rename('Fold', inplace=True)

# Print the evaluation scores table
print(T)
