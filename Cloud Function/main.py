import pandas as pd
import numpy as np
import tensorflow
from tensorflow import keras
import sklearn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM 
from tensorflow.keras.layers import Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from google.cloud import storage
import json
from scipy.io import loadmat
from pymatreader import read_mat
import os
import h5py
import gcsfs
import firebase_admin
from firebase_admin import db
import urllib.request
from scipy.signal import butter, filtfilt, iirnotch, savgol_filter
import math
from numpy import genfromtxt
from tensorflow.keras.models import load_model






def read_mat(mat_file):
    vals = []
    val_dict = loadmat(mat_file)
    vals.append([item for sublist in val_dict['val'] for item in sublist])
    return vals

def bpm(signal, beats):
    print('Signal Length: ' + str(len(signal)))
    total_time = len(signal)*0.003
    total_beats = len(beats)
    #print(total_beats)
    rate_bpm  = (60*total_beats)/total_time
    print('BPM: ' + str(rate_bpm))
    return rate_bpm
    
    


def classify_signal(preds):
    normal_beats = 0
    for i in range(len(preds)):
        if preds[i][0] == 1:
            normal_beats += 1
    if normal_beats/len(preds) > 0.60:
        print('Signal Classification: Normal')
        return 'normal'
    else:
        print('Signal Classification: Abnormal')
        return 'abnormal'
    



def butter_lowpass(cutoff, sample_rate, order=2):
    '''standard lowpass filter.

    Function that defines standard Butterworth lowpass filter

    Parameters
    ----------
    cutoff : int or float
        frequency in Hz that acts as cutoff for filter.
        All frequencies above cutoff are filtered out.

    sample_rate : int or float
        sample rate of the supplied signal

    order : int
        filter order, defines the strength of the roll-off
        around the cutoff frequency. Typically orders above 6
        are not used frequently.
        default: 2
    
    Returns
    -------
    out : tuple
        numerator and denominator (b, a) polynomials
        of the defined Butterworth IIR filter.

    Examples
    --------
    >>> b, a = butter_lowpass(cutoff = 2, sample_rate = 100, order = 2)
    >>> b, a = butter_lowpass(cutoff = 4.5, sample_rate = 12.5, order = 5)
    '''
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_highpass(cutoff, sample_rate, order=2):
    '''standard highpass filter.

    Function that defines standard Butterworth highpass filter

    Parameters
    ----------
    cutoff : int or float
        frequency in Hz that acts as cutoff for filter.
        All frequencies below cutoff are filtered out.

    sample_rate : int or float
        sample rate of the supplied signal

    order : int
        filter order, defines the strength of the roll-off
        around the cutoff frequency. Typically orders above 6
        are not used frequently.
        default : 2
    
    Returns
    -------
    out : tuple
        numerator and denominator (b, a) polynomials
        of the defined Butterworth IIR filter.

    Examples
    --------
    we can specify the cutoff and sample_rate as ints or floats.

    >>> b, a = butter_highpass(cutoff = 2, sample_rate = 100, order = 2)
    >>> b, a = butter_highpass(cutoff = 4.5, sample_rate = 12.5, order = 5)
    '''
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_bandpass(lowcut, highcut, sample_rate, order=2):
    '''standard bandpass filter.
    Function that defines standard Butterworth bandpass filter.
    Filters out frequencies outside the frequency range
    defined by [lowcut, highcut].

    Parameters
    ----------
    lowcut : int or float
        Lower frequency bound of the filter in Hz

    highcut : int or float
        Upper frequency bound of the filter in Hz

    sample_rate : int or float
        sample rate of the supplied signal

    order : int
        filter order, defines the strength of the roll-off
        around the cutoff frequency. Typically orders above 6
        are not used frequently.
        default : 2
    
    Returns
    -------
    out : tuple
        numerator and denominator (b, a) polynomials
        of the defined Butterworth IIR filter.

    Examples
    --------
    we can specify lowcut, highcut and sample_rate as ints or floats.

    >>> b, a = butter_bandpass(lowcut = 1, highcut = 6, sample_rate = 100, order = 2)
    >>> b, a = butter_bandpass(lowcut = 0.4, highcut = 3.7, sample_rate = 72.6, order = 2)
    '''
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def filter_signal(data, cutoff, sample_rate, order=2, filtertype='lowpass',
                  return_top = False):
    '''Apply the specified filter

    Function that applies the specified lowpass, highpass or bandpass filter to
    the provided dataset.

    Parameters
    ----------
    data : 1-dimensional numpy array or list 
        Sequence containing the to be filtered data

    cutoff : int, float or tuple
        the cutoff frequency of the filter. Expects float for low and high types
        and for bandpass filter expects list or array of format [lower_bound, higher_bound]

    sample_rate : int or float
        the sample rate with which the passed data sequence was sampled

    order : int
        the filter order 
        default : 2

    filtertype : str
        The type of filter to use. Available:
        - lowpass : a lowpass butterworth filter
        - highpass : a highpass butterworth filter
        - bandpass : a bandpass butterworth filter
        - notch : a notch filter around specified frequency range
        both the highpass and notch filter are useful for removing baseline wander. The notch
        filter is especially useful for removing baseling wander in ECG signals.


    Returns
    -------
    out : 1d array
        1d array containing the filtered data

    Examples
    --------
    >>> import numpy as np
    >>> import heartpy as hp

    Using standard data provided

    >>> data, _ = hp.load_exampledata(0)

    We can filter the signal, for example with a lowpass cutting out all frequencies
    of 5Hz and greater (with a sloping frequency cutoff)

    >>> filtered = filter_signal(data, cutoff = 5, sample_rate = 100.0, order = 3, filtertype='lowpass')
    >>> print(np.around(filtered[0:6], 3))
    [530.175 517.893 505.768 494.002 482.789 472.315]

    Or we can cut out all frequencies below 0.75Hz with a highpass filter:

    >>> filtered = filter_signal(data, cutoff = 0.75, sample_rate = 100.0, order = 3, filtertype='highpass')
    >>> print(np.around(filtered[0:6], 3))
    [-17.975 -28.271 -38.609 -48.992 -58.422 -67.902]

    Or specify a range (here: 0.75 - 3.5Hz), outside of which all frequencies
    are cut out.

    >>> filtered = filter_signal(data, cutoff = [0.75, 3.5], sample_rate = 100.0, 
    ... order = 3, filtertype='bandpass')
    >>> print(np.around(filtered[0:6], 3))
    [-12.012 -23.159 -34.261 -45.12  -55.541 -65.336]

    A 'Notch' filtertype is also available (see remove_baseline_wander).
    
    >>> filtered = filter_signal(data, cutoff = 0.05, sample_rate = 100.0, filtertype='notch')

    Finally we can use the return_top flag to only return the filter response that
    has amplitute above zero. We're only interested in the peaks, and sometimes
    this can improve peak prediction:

    >>> filtered = filter_signal(data, cutoff = [0.75, 3.5], sample_rate = 100.0, 
    ... order = 3, filtertype='bandpass', return_top = True)
    >>> print(np.around(filtered[48:53], 3))
    [ 0.     0.     0.409 17.088 35.673]
    '''
    if filtertype.lower() == 'lowpass':
        b, a = butter_lowpass(cutoff, sample_rate, order=order)
    elif filtertype.lower() == 'highpass':
        b, a = butter_highpass(cutoff, sample_rate, order=order)
    elif filtertype.lower() == 'bandpass':
        assert type(cutoff) == tuple or list or np.array, 'if bandpass filter is specified, \
cutoff needs to be array or tuple specifying lower and upper bound: [lower, upper].'
        b, a = butter_bandpass(cutoff[0], cutoff[1], sample_rate, order=order)
    elif filtertype.lower() == 'notch':
        b, a = iirnotch(cutoff, Q = 0.005, fs = sample_rate)
    else:
        raise ValueError('filtertype: %s is unknown, available are: \
lowpass, highpass, bandpass, and notch' %filtertype)

    filtered_data = filtfilt(b, a, data)
    
    if return_top:
        return np.clip(filtered_data, a_min = 0, a_max = None)
    else:
        return filtered_data



def remove_baseline_wander(data, sample_rate, cutoff=0.05):
    '''removes baseline wander

    Function that uses a Notch filter to remove baseline
    wander from (especially) ECG signals

    Parameters
    ----------
    data : 1-dimensional numpy array or list 
        Sequence containing the to be filtered data

    sample_rate : int or float
        the sample rate with which the passed data sequence was sampled

    cutoff : int, float 
        the cutoff frequency of the Notch filter. We recommend 0.05Hz.
        default : 0.05

    Returns
    -------
    out : 1d array
        1d array containing the filtered data

    Examples
    --------
    >>> import heartpy as hp
    >>> data, _ = hp.load_exampledata(0)

    baseline wander is removed by calling the function and specifying
    the data and sample rate.

    >>> filtered = remove_baseline_wander(data, 100.0)
    '''

    return filter_signal(data = data, cutoff = cutoff, sample_rate = sample_rate,
                         filtertype='notch')





def quotient_filter(RR_list, RR_list_mask = [], iterations=2):
    '''applies a quotient filter

    Function that applies a quotient filter as described in
    "Piskorki, J., Guzik, P. (2005), Filtering Poincare plots"

    Parameters
    ----------
    RR_list - 1d array or list
        array or list of peak-peak intervals to be filtered

    RR_list_mask - 1d array or list
        array or list containing the mask for which intervals are 
        rejected. If not supplied, it will be generated. Mask is 
        zero for accepted intervals, one for rejected intervals.

    iterations - int
        how many times to apply the quotient filter. Multipled
        iterations have a stronger filtering effect
        default : 2

    Returns
    -------
    RR_list_mask : 1d array
        mask for RR_list, 1 where intervals are rejected, 0 where
        intervals are accepted.

    Examples
    --------
    Given some example data let's generate an RR-list first
    >>> import heartpy as hp
    >>> data, timer = hp.load_exampledata(1)
    >>> sample_rate = hp.get_samplerate_mstimer(timer)
    >>> wd, m = hp.process(data, sample_rate)
    >>> rr = wd['RR_list']
    >>> rr_mask = wd['RR_masklist']

    Given this data we can use this function to further clean the data:
    >>> new_mask = quotient_filter(rr, rr_mask)

    Although specifying the mask is optional, as you may not always have a
    pre-computed mask available:
    >>> new_mask = quotient_filter(rr)
    
    '''

    if len(RR_list_mask) == 0:
        RR_list_mask = np.zeros((len(RR_list)))
    else:
        assert len(RR_list) == len(RR_list_mask), \
        'error: RR_list and RR_list_mask should be same length if RR_list_mask is specified'

    for iteration in range(iterations):
        for i in range(len(RR_list) - 1):
            if RR_list_mask[i] + RR_list_mask[i + 1] != 0:
                pass #skip if one of both intervals is already rejected
            elif 0.8 <= RR_list[i] / RR_list[i + 1] <= 1.2:
                pass #if R-R pair seems ok, do noting
            else: #update mask
                RR_list_mask[i] = 1
                #RR_list_mask[i + 1] = 1

    return np.asarray(RR_list_mask)


def smooth_signal(data, sample_rate, window_length=None, polyorder=3):
    '''smooths given signal using savitzky-golay filter

    Function that smooths data using savitzky-golay filter using default settings.

    Functionality requested by Eirik Svendsen. Added since 1.2.4

    Parameters
    ----------
    data : 1d array or list
        array or list containing the data to be filtered

    sample_rate : int or float
        the sample rate with which data is sampled

    window_length : int or None
        window length parameter for savitzky-golay filter, see Scipy.signal.savgol_filter docs.
        Must be odd, if an even int is given, one will be added to make it uneven.
        default : 0.1  * sample_rate

    polyorder : int
        the order of the polynomial fitted to the signal. See scipy.signal.savgol_filter docs.
        default : 3

    Returns
    -------
    smoothed : 1d array
        array containing the smoothed data

    Examples
    --------
    Given a fictional signal, a smoothed signal can be obtained by smooth_signal():

    >>> x = [1, 3, 4, 5, 6, 7, 5, 3, 1, 1]
    >>> smoothed = smooth_signal(x, sample_rate = 2, window_length=4, polyorder=2)
    >>> np.around(smoothed[0:4], 3)
    array([1.114, 2.743, 4.086, 5.   ])

    If you don't specify the window_length, it is computed to be 10% of the 
    sample rate (+1 if needed to make odd)
    >>> import heartpy as hp
    >>> data, timer = hp.load_exampledata(0)
    >>> smoothed = smooth_signal(data, sample_rate = 100)

    '''

    if window_length == None:
        window_length = sample_rate // 10
        
    if window_length % 2 == 0 or window_length == 0: window_length += 1

    smoothed = savgol_filter(data, window_length = window_length,
                             polyorder = polyorder)

    return smoothed


def rwb_filter(signals):
    fltr_signal = []
    for i in range(len(signals)):
        fltr_signal.append(remove_baseline_wander(signals[i], 300))
    return fltr_signal



def read_ecg(file_name):
    return genfromtxt(file_name, delimiter=',')

def lgth_transform(ecg, ws):
    lgth=ecg.shape[0]
    sqr_diff=np.zeros(lgth)
    diff=np.zeros(lgth)
    ecg=np.pad(ecg, ws, 'edge')
    for i in range(lgth):
        temp=ecg[i:i+ws+ws+1]
        left=temp[ws]-temp[0]
        right=temp[ws]-temp[-1]
        diff[i]=min(left, right)
        diff[diff<0]=0
    # sqr_diff=np.multiply(diff, diff)
    # diff=ecg[:-1]-ecg[1:]
    # sqr_diff[:-1]=np.multiply(diff, diff)
    # sqr_diff[-1]=sqr_diff[-2]
    return np.multiply(diff, diff)

def integrate(ecg, ws):
    lgth=ecg.shape[0]
    integrate_ecg=np.zeros(lgth)
    ecg=np.pad(ecg, math.ceil(ws/2), mode='symmetric')
    for i in range(lgth):
        integrate_ecg[i]=np.sum(ecg[i:i+ws])/ws
    return integrate_ecg

def find_peak(data, ws):
    lgth=data.shape[0]
    true_peaks=list()
    for i in range(lgth-ws+1):
        temp=data[i:i+ws]
        if np.var(temp)<5:
            continue
        index=int((ws-1)/2)
        peak=True
        for j in range(index):
            if temp[index-j]<=temp[index-j-1] or temp[index+j]<=temp[index+j+1]:
                peak=False
                break

        if peak is True:
            true_peaks.append(int(i+(ws-1)/2))
    return np.asarray(true_peaks)

def find_R_peaks(ecg, peaks, ws):
    num_peak=peaks.shape[0]
    R_peaks=list()
    for index in range(num_peak):
        i=peaks[index]
        if i-2*ws>0 and i<ecg.shape[0]:
            temp_ecg=ecg[i-2*ws:i]
            R_peaks.append(int(np.argmax(temp_ecg)+i-2*ws))
    #print('R-peak function')
    #print(R_peaks)
    return np.asarray(R_peaks)

def find_S_point(ecg, R_peaks):
    num_peak=R_peaks.shape[0]
    S_point=list()
    for index in range(num_peak):
        i=R_peaks[index]
        cnt=i
        if cnt+1>=ecg.shape[0]:
            break
        while ecg[cnt]>ecg[cnt+1]:
            cnt+=1
            if cnt>=ecg.shape[0]:
                break
        S_point.append(cnt)
    return np.asarray(S_point)


def find_Q_point(ecg, R_peaks):
    num_peak=R_peaks.shape[0]
    Q_point=list()
    for index in range(num_peak):
        i=R_peaks[index]
        cnt=i
        if cnt-1<0:
            break
        while ecg[cnt]>ecg[cnt-1]:
            cnt-=1
            if cnt<0:
                break
        Q_point.append(cnt)
    return np.asarray(Q_point)

def EKG_QRS_detect(ecg, fs, QS, plot=False):
    sig_lgth=ecg.shape[0]
    ecg=ecg-np.mean(ecg)
    ecg_lgth_transform=lgth_transform(ecg, int(fs/20))
    # ecg_lgth_transform=lgth_transform(ecg_lgth_transform, int(fs/40))

    ws=int(fs/8)
    ecg_integrate=integrate(ecg_lgth_transform, ws)/ws
    ws=int(fs/6)
    ecg_integrate=integrate(ecg_integrate, ws)
    ws=int(fs/36)
    ecg_integrate=integrate(ecg_integrate, ws)
    ws=int(fs/72)
    ecg_integrate=integrate(ecg_integrate, ws)

    peaks=find_peak(ecg_integrate, int(fs/10))
    R_peaks=find_R_peaks(ecg, peaks, int(fs/40))
    if QS:
        S_point=find_S_point(ecg, R_peaks)
        Q_point=find_Q_point(ecg, R_peaks)
    else:
        S_point=None
        Q_point=None
    if plot:
        index=np.arange(sig_lgth)/fs
        fig, ax=plt.subplots()
        ax.plot(index, ecg, 'b', label='ECG')
        ax.plot(R_peaks/fs, ecg[R_peaks], 'ro', label='R peaks')
        if QS:
            ax.plot(S_point/fs, ecg[S_point], 'go', label='S')
            ax.plot(Q_point/fs, ecg[Q_point], 'yo', label='Q')
        ax.set_xlim([0, sig_lgth/fs])
        ax.set_xlabel('Time [sec]')
        ax.legend()
        # ax[1].plot(ecg_integrate)
        # ax[1].set_xlim([0, ecg_integrate.shape[0]])
        # ax[2].plot(ecg_lgth_transform)
        # ax[2].set_xlim([0, ecg_lgth_transform.shape[0]])
        #fig.figure(figsize=(20,10))
        plt.show()
    return R_peaks, S_point, Q_point


def QRS(data_list):
    '''
    QRS detection on 1-D numpy array
    300 Hz sampling rate
    '''
    fs=300
    R_peaks, S_point, Q_point=EKG_QRS_detect(np.asarray(data_list), fs, False)
    print(R_peaks)
    return R_peaks


def signal_peaks(signals):
    peaks = []
    for i in range(len(signals)):
        peaks.append(QRS(signals[i]))
    return peaks  


def slice_signal(rpeaks, signal):
    signal_beats = []

    inx = 0
    if len(rpeaks) >= 10:
        for i in range(len(rpeaks)):
            #print('inx: ' + str(inx))
            #print('R-Peak Position: ' + str(rpeaks[i]))
            signal_beats.append(signal[inx:rpeaks[i]+1])
            #beat_lengths.append()
            inx = rpeaks[i]+1
        
    return signal_beats


def slicer(full_signal, signal_peaks):
    beats = []
    for i in range(len(full_signal)):
        beats_arr = slice_signal(signal_peaks[i], full_signal[i])
        for j in range(len(beats_arr)):
            if len(beats_arr[j]) > 0:
                beats.append(beats_arr[j])
    return beats


def trim_outliers(beats):
    trimmed = []
    for i in range(len(beats)):
        if len(beats[i]) > 71 and len(beats[i]) < 420:
            trimmed.append(beats[i])
            
    #padding with zeroes to make beats of equal length
    padded = np.asarray([np.pad(a, (0, 419 - len(a)), 'constant', constant_values=0) for a in trimmed])
    
    return padded


def normalize(dataframe):
    dataset = dataframe.values
    X = dataset[:,0:419].astype(float)
    scaler = MinMaxScaler()
    X_norm = pd.DataFrame(scaler.fit_transform(X))
    X_norm_vals = X_norm.values
    #print(X[1])
    #print(X_norm_vals[1])
    print(len(X_norm_vals[1]))
    print(len(X[1]))
    
    X_adj_dim = np.reshape(X_norm_vals, (X_norm_vals.shape[0], 1, X_norm_vals.shape[1]))
    
    return X_adj_dim






def get_predictions(input_data):
    fs = gcsfs.GCSFileSystem(project='fit3162-cardio')
    print(fs.ls('saved-cardio-models/model')) 

    with fs.open('saved-cardio-models/model/model2.h5', 'rb') as model_file:
        model_gcs = h5py.File(model_file, 'r')
        print('Loading Classifier.....')
        saved_lstm_rnn = tensorflow.keras.models.load_model(model_gcs, compile = True)
        Xnew = input_data
        ynew = (saved_lstm_rnn.predict(Xnew) > 0.5).astype(int)    #saved_lstm_rnn.predict(Xnew)
        # show the inputs and predicted outputs
        for i in range(len(Xnew)):
            print("Predicted=%s" % (ynew[i]))    #X=%s, Xnew[i], 
        return ynew













def load_signal(sig_file_name):
    fs = gcsfs.GCSFileSystem(project='fit3162-cardio')
    signal_file = 'saved-cardio-models/' + str(sig_file_name) 
    with fs.open(signal_file, 'rb') as model_file:
        signal_data = read_mat(model_file)
        rwb_fltrd_signal = rwb_filter(signal_data)
        signal_rpkeaks = signal_peaks(rwb_fltrd_signal)
        beats = slicer(rwb_fltrd_signal, signal_rpkeaks)
        heart_rate = bpm(signal_data[0], beats)
        
        padded_beats = trim_outliers(beats)
        pred_df = pd.DataFrame(padded_beats)
        
        input_data_norm = normalize(pred_df)
        
        predictions = get_predictions(input_data_norm)
        
        signal_class = classify_signal(predictions)
        
        uploadClassificationResultCloud(signal_class)
        
        uploadHeartRateCloud(heart_rate)
        
        db.reference("users/4BxcQ1NPHOUVGcICDK2EBffMzfv2/").update({"model_processing": "False"})





def firebase_connect():
    fs = gcsfs.GCSFileSystem(project='fit3162-cardio')
    print('Connecting to Firebase...')
    with fs.open('firebase-cred/cardiomobilefyp-firebase-adminsdk-hekbv-8310eee5ad.json', 'rb') as cred_file:
        cred_dic = json.load(cred_file)
        cred_obj = firebase_admin.credentials.Certificate(cred_dic)
        default_app = firebase_admin.initialize_app(cred_obj, {
            'databaseURL': "https://cardiomobilefyp.firebaseio.com/"
            })
        




def uploadClassificationResultCloud(model_result):
    path = "users/4BxcQ1NPHOUVGcICDK2EBffMzfv2/"
    db.reference(path).update({"classification_result" : model_result})

def uploadHeartRateCloud(ave_hr):
    path = "users/4BxcQ1NPHOUVGcICDK2EBffMzfv2/"
    db.reference(path).update({"heart_rate": ave_hr}) 







def hello_gcs(event, context):
     """Triggered by a change to a Cloud Storage bucket.
     Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
     """
     try:
        firebase_connect()
        file = event
        print(f"Processing file: {file['name']}")
        load_signal(file['name'])
     except:
         print('Firebase Connection Already Established')
         file = event
         print(f"Processing file: {file['name']}")
         load_signal(file['name'])