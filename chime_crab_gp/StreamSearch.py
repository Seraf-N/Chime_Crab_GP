import numpy as np

import os

import astropy.units as u
from astropy.time import Time
from astropy.table import Table

import time

workdir = '/mnt/scratch-lustre/nadeau/Chime/'
codedir = '/mnt/scratch-lustre/nadeau/Chime/Code'
banddir = '/mnt/scratch-lustre/hhlin/Data/20181019T113802Z_chime_psr_vdif'
plotdir = '/mnt/scratch-lustre/nadeau/Chime/Code/Plots'
tempdir = '/mnt/scratch-lustre/nadeau/Chime/Data/20181019T113802Z_temp_vdif'

splitdir= '/mnt/scratch-lustre/nadeau/Chime/Data/20181019T113802Z_split_vdif'
istream = '/mnt/scratch-lustre/nadeau/Chime/Data/20181019T113802Z_i_stream'

os.chdir(istream)

start_time = Time('2018-10-19T11:38:01.507200000', format='isot', precision=9)

def stream_run(n, im, time_waste, w=31250, dm=56.61, s_per_sample=2.56e-4):
    
    sample_min = w*n - time_waste*n
    sample_max = sample_min + w
    
    MAX = len(I)
    if sample_min > MAX:
        return 'end'
    elif sample_max > MAX:
        sample_max = MAX
    
    print(sample_min, sample_max)
    
    IM = im[:, sample_min:sample_max]
    
    #####################################################################################

    t0 = time.time()
    dm = DispersionMeasure(56.61)
    dt = dm.time_delay(800*u.MHz, np.linspace(800, 400, 1024)*u.MHz)
    
    ddim = imshift(IM*1, shiftc=dt.value/s_per_sample)
    print('Image dedispersed, t={}s'.format(time.time()-t0))

    I_test = np.nanmean(ddim, axis=0)
    mean_test = np.nanmean(I_test)
    std_test = np.nanstd(I_test)
    snr_test = (I_test-mean_test)/std_test
    
    ######################################################################################################
    
    time_off = sample_min * s_per_sample
    
    ######################################################################################################
    
    N=300
    rollmean = np.convolve(snr_test, np.ones((N,))/N)[(N-1):]
    
    ######################################################################################################
    
    snr_cut = (snr_test - rollmean)[:len(snr_test)-int(time_waste)] 
    pos = np.argmax(snr_cut)
    snr = np.max(snr_cut)
    
    #return [sample_min, sample_max, pos, snr, snr_cut]
    
    return ddim, I_test


samples_per_frame = 15625
n_frames = 25*320
binning = 100
s_per_sample = 2.56e-6 * binning

I = np.memmap('i_stream.dat', dtype=np.float32, mode='r', shape=(int(samples_per_frame*n_frames/100), 1024))


##################################################################################
# Masking of intensity stream
# Filling masked portions with mean value of each frequency channel
##################################################################################

IM = I.transpose(1,0)
im = IM * 1

t0 = time.time()

for i in range(len(im)): # Loop over frequency channels
    row2 = im[i]*1
    m = im[i] == 0
    if np.sum(m) != 0:
        row2[m] = np.nan        # Mask all true zeros to nan
        mean = np.nanmean(row2) # Compute the mean of each channel
        if np.isnan(mean):
            im[i][m] = 0 # if channel mean is nan, then fill channel back with 0s
        else:
            im[i][m] = mean # Fill gaps in channel with the channel mean value
    else:
        im[i] = np.zeros(np.shape(im[i]))
    print('{}/{}: {}% complete'.format(i+1, len(im), 100*(i+1)/len(im)), end='                                  \r')
        
tf = time.time()
print('Masking complete: ', tf-t0, end=' s                                                                          \n')        
    
##################################################################################
# Dedispersing the intensity stream
##################################################################################

# Obtains snr data for w samples, selected based on n
dm = 56.61
dm = DispersionMeasure(dm)
dt = dm.time_delay(800*u.MHz, 400*u.MHz)
    
# Convert dt to samples
time_waste = int(abs(dt.value / s_per_sample) + 1)
print(time_waste, ' samples lost at the end of array due to dedispersion')

w = 31250 

ddim = np.zeros(np.shape(im))
i = np.zeros(len(I))

sample = 0

T0 = time.time()

for n in range(len(I)):
    
    print(n)
    
    im_out = stream_run(n, im, time_waste, w, dm=dm, s_per_sample=s_per_sample)
    
    if im_out == 'end':
        break
        
    d_sample = len(im_out[1]) - time_waste
    
    x, y = im_out
    
    ddim[:, sample:sample+d_sample] = x[:,:d_sample]
    i[sample:sample+d_sample] = y[:d_sample]
    
    sample += d_sample

#'''
TF = time.time()

T = TF - T0
hh = int(T/3600)
mm = int((T % 3600)/60)
ss = int(T % 60)
print('Time for completion: {}h{}m{}s'.format(hh, mm, ss))


############################################################################
# Apply corrections to intensity stream background itteratively
############################################################################

iter = 4
N = 300
buffer = 5000

snr_test = i * 1

for _ in range(iter):
    
    rollmean = np.convolve(snr_test, np.ones((N,))/N)[(N-1):]
    snr_test = snr_test - rollmean
    
mean_test = np.nanmean(snr_test)
std_test = np.nanstd(snr_test)

snr_test = (snr_test - mean_test) / std_test

#############################################################################
# Search intensity stream for Giant Pulses
#############################################################################

snr_search = snr_test * 1

cutoff = 4 # set S/N cutoff

searching = True

POS = []
SNR = []

pos = np.argmax(snr_search)
snr = snr_search[pos]

while snr > cutoff:
    
    POS += [pos]
    SNR += [snr]
    
    snr_search[pos-10:pos+10] = 0
    
    pos = np.argmax(snr_search)
    snr = snr_search[pos]
    
POS = np.array(POS)
TIME_S = POS * s_per_sample
SNR = np.array(SNR)
MJD = start_time + TIME_S

#############################################################################
# Create Table of GPs to be saved
#############################################################################

tab = Table()

tab['mjd'] = MJD
tab['time'] = TIME
tab['pos'] = POS
tab['snr'] = SNR
tab.sort('mjd')

os.chdir(istream)
overwrite=False
tab.write('Crab_GP_tab-{}_sigma-binning_{}.fits'.format(cutoff, binning), overwrite=overwrite)



