import numpy as np

import os
import baseband
import scintillometry
from baseband import vdif

import astropy.units as u
#from astropy.time import Time
#from astropy.table import Table

from scintillometry.dispersion import Dedisperse
from scintillometry.shaping import ChangeSampleShape
from scintillometry.fourier import get_fft_maker
from scintillometry.combining import Concatenate

from scipy.stats import binned_statistic as bs

import time
from functools import reduce

workdir = '/mnt/scratch-lustre/nadeau/Chime/'
codedir = '/mnt/scratch-lustre/nadeau/Chime/Code'
banddir = '/mnt/scratch-lustre/hhlin/Data/20181019T113802Z_chime_psr_vdif'
plotdir = '/mnt/scratch-lustre/nadeau/Chime/Code/Plots'

splitdir= '/mnt/scratch-lustre/nadeau/Chime/Data/20181019T113802Z_split_vdif'
tempdir = '/mnt/scratch-lustre/nadeau/Chime/Data/20181019T113802Z_temp_vdif'
chandir = '/mnt/scratch-lustre/nadeau/Chime/Data/20181019T113802Z_chan_vdif'
dispdir = '/mnt/scratch-lustre/nadeau/Chime/Data/20181019T113802Z_disp_vdif'
datadir = '/mnt/scratch-lustre/nadeau/Chime/Data/'

pulsedir = '/mnt/scratch-lustre/nadeau/Chime/Data/20181019T113802Z_pulse_vdif'

os.chdir(splitdir)

x = os.listdir(tempdir)
x.sort()

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

dm = 56.61 * u.pc / u.cm**3 # Set up the dispersion measure.

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

os.chdir(istream)
tab = Table.read('Crab_GP_tab-4_sigma-binning_100.fits')

POS = tab['pos']

for pos in POS:

    t0 = time.time()

    pulsechans = np.zeros((38750, 1024, 2), dtype=np.complex)
    start_time0 = out_frames[0].start_time

    i = 0

    for frame in out_frames:

        pulse_time = (pos - 13000)*2.56*u.us
        dt = frame.start_time - start_time0

        frame.seek(pulse_time - dt)
        pulse = frame.read(38750)

        pulsechans[:, i:i+8, :] = pulse#imbin(pulse[:,np.newaxis], binning_c=100)

        i += 8

        print('Channels {}-{} loaded: {}% Complete'.format(i, i+8, 100*(i+8)/1024), end='                 \r')

    print('Reading: ', time.time() - t0, end='                    ')
    
    # CODE TO WRITE EACH PULSE BB BEFORE MOVING ON TO THE NEXT PULSE

    
    