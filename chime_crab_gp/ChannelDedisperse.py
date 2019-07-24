import numpy as np

import os
import baseband
import scintillometry
from baseband import vdif

import astropy.units as u
from astropy.time import Time
from astropy.table import QTable

from scintillometry.fourier import NumpyFFTMaker, get_fft_maker
get_fft_maker.default = NumpyFFTMaker()

from scintillometry.dispersion import Dedisperse
from scintillometry.shaping import ChangeSampleShape
from scintillometry.fourier import get_fft_maker
from scintillometry.combining import Concatenate

from scipy.stats import binned_statistic as bs

from numpy.lib.format import open_memmap

import time
from functools import reduce

'''# CITA WORK DIRECTORIES
workdir = '/mnt/scratch-lustre/nadeau/Chime/'
codedir = '/mnt/scratch-lustre/nadeau/Chime/Code'
banddir = '/mnt/scratch-lustre/hhlin/Data/20181019T113802Z_chime_psr_vdif'
plotdir = '/mnt/scratch-lustre/nadeau/Chime/Code/Plots'

tempdir = '/mnt/scratch-lustre/nadeau/Chime/Data/20181019T113802Z_temp_vdif'
chandir = '/mnt/scratch-lustre/nadeau/Chime/Data/20181019T113802Z_chan_vdif'
dispdir = '/mnt/scratch-lustre/nadeau/Chime/Data/20181019T113802Z_disp_vdif'
datadir = '/mnt/scratch-lustre/nadeau/Chime/Data/'

pulsedir = '/mnt/scratch-lustre/nadeau/Chime/Data/pulse_vdif'
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
pulsedir = testdir + f'{datestr}/pulsedir/'

# Open and store pointers to input data files, written at the end of ChannelSplit
os.chdir(splitdir)  
in_frames = []
for i in range(0, 1024, 8):
    
    fname = 'Split_Channel_c{:04d}-{:04d}.vdif'.format(i, i+7)
    chanframe = vdif.open(fname, 'rs', sample_rate=1/(2.56*u.us))
    in_frames.append(chanframe)

frequency = (600
             - 400 / 2.  # Down to bottom end
             + 400/1024 / 2.  # Up to center of channel 0
             + 400/1024 * np.arange(1024)) * u.MHz # CHIME has 1024 channels

#For chime, index 0 of frequency axis is 800MHz and index -1 is 400MHz
frequency = frequency[::-1]

# Form to have the correct shape for the dedisperse function to operate
freq = np.zeros((1024, 2))
freq[:, 0] = frequency
freq[:, 1] = frequency

splittab = QTable.read(f'{istream}SplitTab.fits')

sigma = 3.5
binning = splittab.meta['I_BIN']
mjd = Time(splittab.meta['T_START'], format='isot', precision=9).mjd
tab = QTable.read(f'{istream}GP_tab-mjd_{mjd:.2f}-sigma_{sigma}-binning_{binning}.fits')

tbin = splittab.meta['TBIN'] * u.s
dm = tab.meta['DM'] * u.pc / u.cm**3 # Set up the dispersion measure.

def reshape(data):
    reshaped = data.transpose(0, 2, 1)
    return reshaped

out_frames = []
t0 = time.time()
frames = []
i=0

for frame in in_frames:
    
    chanframe = ChangeSampleShape(frame, reshape)
    frames += [chanframe]
    f = freq[i:i+8]
    dedisperse = Dedisperse(chanframe, dm, frequency=f*u.MHz, 
                            sideband=1, reference_frequency=800*u.MHz)
    
    out_frames += [dedisperse]
    i += 8

print('Dedispersion: ', time.time() - t0)

T_START = time.time()

tab.sort('snr')
tab.reverse()

m = tab['pos'] > 600
tab = tab[m]
m = tab['pos'] < tab.meta['NSAMPLES'] / binning - 5000 # ~ Dedispersion time + 2 pulse periods from edge
tab = tab[m]

POS = tab['pos']

FNAMES = []

START_TIMES = []

id = 0
prepulse = 2625
pulse_width = 15625

t0 = time.time()

for pos in POS:

    fname = 'pulse_{:04d}.npy'.format(id)
    FNAMES += [fname]
    
    os.chdir(pulsedir)
    pulsechans = open_memmap(fname, dtype=np.float16, mode='w+', shape=(pulse_width, 1024, 4))#open_memmap(fname, dtype=np.complex64, mode='w+', shape=(pulse_width, 1024, 2))
    
    start_time0 = out_frames[0].start_time

    start_times = []
    i = 0

    pulse = np.zeros((pulse_width, 1024, 4), dtype=np.float16)#2), dtype=np.complex64)
    for frame in out_frames:

        pulse_time = (pos * binning - prepulse) * tbin
        dt = frame.start_time - start_time0
        
        for _ in range(8):
            chantime = frame.start_time + (pos * binning - prepulse) * tbin
            start_times += [chantime.isot]
            
        frame.seek(pulse_time - dt)
        readpulse = frame.read(pulse_width)        

        #pulse[:,i:i+8,:] = frame.read(pulse_width)
        pulse[:,i:i+8,0] = np.real(readpulse[:,:,0])
        pulse[:,i:i+8,1] = np.imag(readpulse[:,:,0])
        pulse[:,i:i+8,2] = np.real(readpulse[:,:,1])
        pulse[:,i:i+8,3] = np.imag(readpulse[:,:,1])        

        #print('Channels {}-{} loaded: {}% Complete'.format(i, i+8, 100*(i+8)/1024), end='                 \r')
        i += 8
        
    pulsechans[:] = pulse

    #print('Reading: ', time.time() - t0, end='                    ')
    tf = time.time()
    print(f'Pulses written: {id+1}/{len(POS)} - {100*(id+1)/len(POS):.2f}% complete - {tf-t0}s elapsed', end='                \r')
    
    START_TIMES += [start_times]
    id += 1
    
    if id > 999:
        print('Pulse limit reached: 1000 pulses already saved for this dataset.')
        break

        
tab['fname'] = FNAMES
times = Time(START_TIMES) + (pos*binning - prepulse) * tbin
tab['chan_start_times'] = times

tab.meta['freqs'] = frequency.value
tab.meta['freq_u'] = 'MHz'
tab.meta['tbin'] = f'{tbin.value} s'
from collections import namedtuple
tup = namedtuple('Shape', ['time', 'freq', 'pol'])
x = tup(pulse_width, 1024, 4)#2)
tab.meta['shape'] = (pulse_width, 1024, 4)#x
tab.meta['axisname'] = '(time, freq, pol)'
tab.meta['reffreq'] = '800 MHz'
tab.meta['sideband'] = 1
tab.meta['pol'] = 'Re(X)Im(X)Re(Y)Im(Y)'#'XY'
tab.meta['HISTORY'] += ['Outputs vdif of splitchannel dedispersed and pulses identified with StreamSearch cut out and saved as open_memmap objects.']

os.chdir(istream)
tab.write('pulse_tab.fits', overwrite=True)

T_END = time.time()

print('\n {} Pulses written - Time elapsed: {}s'.format(len(POS), T_END-T_START))


