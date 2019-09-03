import numpy as np

import os

import astropy.units as u
from astropy.time import Time
from astropy.table import QTable

from numpy.lib.format import open_memmap
from baseband import vdif

import time

from utils import DispersionMeasure, imshift

'''# CITA DIRECTORIES WORKSPACE
workdir = '/mnt/scratch-lustre/nadeau/Chime/'
codedir = '/mnt/scratch-lustre/nadeau/Chime/Code'
banddir = '/mnt/scratch-lustre/hhlin/Data/20181019T113802Z_chime_psr_vdif'
plotdir = '/mnt/scratch-lustre/nadeau/Chime/Code/Plots'
tempdir = '/mnt/scratch-lustre/nadeau/Chime/Data/20181019T113802Z_temp_vdif'

splitdir= '/mnt/scratch-lustre/nadeau/Chime/Data/20181019T113802Z_split_vdif'
istream = '/mnt/scratch-lustre/nadeau/Chime/Data/20181019T113802Z_i_stream'
'''

# CHIME DIRECTORIES WORKSPACE
import sys
try:
    datestr = sys.argv[1]#'20190626'
    timestr = sys.argv[2]#'191438'
except:
    raise Exception(f'sys.argv has length {len(sys.argv)}. datestr and timestr for dataset not set')

codedir = '/home/serafinnadeau/Scripts/Chime_Crab_GP/chime_crab_gp/'
banddir = '/drives/CHA/'

testdir = '/pulsar-baseband-archiver/crab_gp_archive/'
splitdir = testdir + f'{datestr}/splitdir/'
istream = testdir + f'{datestr}/istream/'

os.chdir(istream)

#start_time = Time('2018-10-19T11:38:01.507200000', format='isot', precision=9)

def stream_run(n, im, time_waste, w=31250, dm=56.61, s_per_sample=2.56e-4):
    
    sample_min = w*n - time_waste*n
    sample_max = sample_min + w
    
    MAX = len(I)#int(samples_per_frame*n_frames/binning)
    if sample_min > MAX:
        return 'end'
    elif sample_max > MAX:
        sample_max = MAX
    
    #print(sample_min, sample_max)
    
    IM = im[:, sample_min:sample_max]
    
    #####################################################################################

    t0 = time.time()
    dm = DispersionMeasure(56.61)
    dt = dm.time_delay(800*u.MHz, np.linspace(800, 400, 1024)*u.MHz)
    
    ddim = imshift(IM*1, shiftc=dt.value/s_per_sample)
    #print('Image dedispersed, t={}s'.format(time.time()-t0), end='                                      \r')

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

splittab = QTable.read('SplitTab.fits')

samples_per_frame = splittab.meta['FRAMELEN']
n_frames = splittab.meta['NFRAMES']
binning = splittab.meta['I_BIN']
s_per_sample = splittab.meta['TBIN'] * binning

t0 = time.time()
I = open_memmap('i_stream.npy', dtype=np.float32, mode='r', 
                shape=(int(samples_per_frame*n_frames/binning), 1024))
tf = time.time()
print(f'Intensity stream loaded')

##################################################################################
# Masking of intensity stream
# Filling masked portions with mean value of each frequency channel
##################################################################################

im = I.transpose(1,0)*1

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
    print('Masking {}/{}: {}% complete'.format(i+1, len(im), 100*(i+1)/len(im)), end='                                  \r')
        
tf = time.time()
print('Masking complete: {:.2f}'.format((tf-t0)/60), end=' min                                                     \n')        
    
##################################################################################
# Dedispersing the intensity stream
##################################################################################

# Obtains snr data for w samples, selected based on n
dm = 56.7
dm = DispersionMeasure(dm)
dt = dm.time_delay(800*u.MHz, 400*u.MHz)
    
# Convert dt to samples
time_waste = int(abs(dt.value / s_per_sample) + 1)
print(time_waste, ' samples lost at the end of array due to dedispersion')

w = 31250 
w_eff = w - time_waste

N = int(len(I) / w_eff) + 1

ddim = np.zeros(np.shape(im))
i = np.zeros(len(I))

sample = 0

T0 = time.time()

for n in range(len(I)):
    
    im_out = stream_run(n, im, time_waste, w, dm=dm, s_per_sample=s_per_sample)
    
    if im_out == 'end':
        break
        
    d_sample = len(im_out[1]) - time_waste

    if d_sample <= 0:
        break
    
    x, y = im_out
    
    ddim[:, sample:sample+d_sample] = x[:,:d_sample]
    i[sample:sample+d_sample] = y[:d_sample]
    
    sample += d_sample
    
    TF = time.time()
    T = TF - T0
    hh = int(T/3600)
    mm = int((T % 3600)/60)
    ss = int(T % 60)
    print(f'Samples {n*w_eff:07d}-{w_eff*(n+1):07d} searched -- Searching {100*(n+1)/N:.2f}% complete -- {hh:02d}h{mm:02d}m{ss:02d}s elapsed', end='  \r')

del im

TF = time.time()

T = TF - T0
hh = int(T/3600)
mm = int((T % 3600)/60)
ss = int(T % 60)
print('Time for completion: {:02d}h{:02d}m{:02d}s'.format(hh, mm, ss))


############################################################################
# Apply corrections to intensity stream background itteratively
############################################################################

N=300

mean_test = np.nanmean(i[:-5000])
std_test = np.nanstd(i[:-5000])
snr_test = (i[:-5000]-mean_test)/std_test

rollmean = np.convolve(snr_test, np.ones((N,))/N)[(N-1):]

snr_test_2 = snr_test - rollmean
mean_test_2 = np.nanmean(snr_test_2)
std_test_2 = np.nanstd(snr_test_2)
snr_test_2 = (snr_test_2-mean_test_2)/std_test_2

rollmean_2 = np.convolve(snr_test_2, np.ones((N,))/N)[(N-1):]

snr_test_3 = snr_test_2 - rollmean_2
mean_test_3 = np.nanmean(snr_test_3)
std_test_3 = np.nanstd(snr_test_3)
snr_test_3 = (snr_test_3-mean_test_3)/std_test_3

rollmean_3 = np.convolve(snr_test_3, np.ones((N,))/N)[(N-1):]

snr_test_4 = snr_test_3 - rollmean_3
mean_test_4 = np.nanmean(snr_test_4)
std_test_4 = np.nanstd(snr_test_4)
snr_test_4 = (snr_test_4-mean_test_4)/std_test_4

rollmean_4 = np.convolve(snr_test_4, np.ones((N,))/N)[(N-1):]

'''
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
'''

print('Corrections applied')

#############################################################################
# Search intensity stream for Giant Pulses
#############################################################################

snr_search = snr_test_4 * 1

istream_corr = open_memmap('istream_corr.npy', dtype=np.float32, mode='w+',
                           shape=np.shape(snr_search))

istream_corr[:] = snr_search

cutoff = 3.5 # set S/N cutoff

searching = True

POS = []
SNR = []
'''
os.chdir(banddir)
x = os.listdir('0/20190626T191438Z_chime_psr_vdif/')
x.sort()
fh_rs = vdif.open('0/20190626T191438Z_chime_psr_vdif/'+x[0], 'rs', sample_rate=1/(2.56*u.us))
start_time = fh_rs.start_time
nsamples = fh_rs.shape[0]
fh_rs.close()

print(start_time)

os.chdir(istream)
'''
start_time = Time(splittab.meta['T_START'], format='isot', precision=9)
nsamples = n_frames * samples_per_frame

pos = np.argmax(snr_search)
snr = snr_search[pos]

while snr > cutoff:
    
    POS += [pos]
    SNR += [snr]
    
    snr_search[pos-10:pos+10] = 0
    
    pos = np.argmax(snr_search)
    snr = snr_search[pos]
    
print(f'Intenisty stream searched for pulses: {len(POS)} pulses found')

POS = np.array(POS)
TIME_S = POS * s_per_sample
SNR = np.array(SNR)
MJD = start_time + TIME_S * u.s

#############################################################################
# Create Table of GPs to be saved
#############################################################################

tab = QTable()

tab['time'] = (TIME_S * u.s + start_time).isot

tab['off_s'] = TIME_S * u.s
tab['pos'] = POS
tab['snr'] = SNR
tab.sort('pos')


tab.meta['DM'] = dm.value 
tab.meta['binning'] = 100
tab.meta['start'] = start_time.isot
tab.meta['nsamples'] = nsamples
tab.meta['history'] = ['Intensity stream i_stream.npy saved from ChannelSplit on vdif files /mnt/scratch-lustre/hhlin/Data/20181019T113802Z_chime_psr_vdif/*', 
                       'i_stream.npy dedispersed and searched for giant pulses']

os.chdir(istream)
tab.write('GP_tab-mjd_{:.2f}-sigma_{}-binning_{}.fits'.format(start_time.mjd, cutoff, binning), overwrite=True)



