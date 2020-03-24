import numpy as np

import os

import sys
if '/home/serafinnadeau/Python/packages/scintillometry/' not in sys.path:
    sys.path.append('/home/serafinnadeau/Python/packages/scintillometry/')

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
banddir = '/drives/STOPGAP/'

#testdir = '/pulsar-baseband-archiver/crab_gp_archive/'
testdir = '/drives/STOPGAP/9/crab_archive/'
splitdir = testdir + f'{datestr}/splitdir/'
istream = testdir + f'{datestr}/istream/'

#########################################################################################
# Open 2D intensity stream memmap object
#########################################################################################

os.chdir(istream)

splittab = QTable.read(istream+'SplitTab.fits')

samples_per_frame = splittab.meta['FRAMELEN']
n_frames = splittab.meta['NFRAMES']
binning = splittab.meta['I_BIN']
s_per_sample = splittab.meta['TBIN'] * binning

I = open_memmap('i_stream.npy', dtype=np.float32, mode='r', 
                shape=(int(samples_per_frame*n_frames/binning), 1024))

print('Intensity stream loaded')

#########################################################################################
# Open, dedisperse, collapse memmap stream chunk by chunk
#########################################################################################

from StreamSearch_utils import *
from Analysis_tools import *

w = 31250 # Chunk width
try:
    tab = QTable.read(istream+'search_tab.fits')
    dm = tab.meta['DM_GUESS']
    guess = False
    print(f'Previous run gives a DM guess of {dm}')
except:
    dm = 56.74 # Stock DM for Crab
    guess = True
    print(f'No previous run: DM guess initially set to {dm}')

stream1D = master(I, w=w, dm=dm, s_per_sample=s_per_sample)

#########################################################################################
# Correct 1D time stream for variations in background levels
#########################################################################################

stream1D_corr = correct_stream(stream1D, istream, N=300)

print('Corrections applied')

#############################################################################
# Search intensity stream for Giant Pulses
#############################################################################

sigma = 2.5
tab = streamsearch(stream1D_corr, splittab, sigma, banddir, istream, datestr, timestr, 
                   Nmax=False, Nmin=1024, dm=DispersionMeasure(dm), output=True)

nu, nu_err = crab_nu_fit(tab, N=100)
nu, nu_err = crab_nu_fit(tab, N=100, start=nu-nu_err, end=nu+nu_err)

time = tab['off_s']
phase = time.value % (1/nu) * nu
component = np.zeros(len(tab))

h = plt.hist(phase, bins=100)
c, b = h[0], h[1]

i = np.argmax(c)

if i-2 >= 0:
    if i+3 < len(b):
        mask_mp = (phase < b[i-2]) + (phase > b[i+3])
    else:
        mask_mp = (phase > b[(i+3) % len(b)]) * (phase < b[i-2])
else:
    mask_mp = (phase < b[i-2]) * (phase > b[i+3])

component[~mask_mp] = 1
pcopy = phase[mask_mp]

h = plt.hist(pcopy, bins=100)
c, b = h[0], h[1]

i = np.argmax(c)

if i-2 >= 0:
    if i+3 < len(b):
        mask_ip = (phase < b[i-2]) + (phase > b[i+3])
    else:
        mask_ip = (phase > b[(i+3) % len(b)]) * (phase < b[i-2])
else:
    mask_ip = (phase < b[i-2]) * (phase > b[i+3])

component[~mask_ip] = 2
pcopy = phase[mask_ip * mask_mp]


tab['phase'] = phase
tab['component'] = component
tab.meta['nu'] = nu
tab.meta['nu_err'] = nu_err


h = plt.hist(pcopy, bins=100)

plt.figure()
plt.plot(tab['phase'], tab['snr'], '.')

samples_per_frame = splittab.meta['FRAMELEN']
n_frames = splittab.meta['NFRAMES']
binning = splittab.meta['I_BIN']
s_per_sample = splittab.meta['TBIN'] * binning


if guess:
    if '/home/serafinnadeau/Python/packages/scintillometry/' not in sys.path:
        sys.path.append('/home/serafinnadeau/Python/packages/scintillometry/')
        
    import scintillometry

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
                 + 400/1024 * np.arange(1024))[::-1]  #u.MHz # CHIME has 1024 channels
    #For chime, index 0 of frequency axis is 800MHz and index -1 is 400MHz

    frequency = np.linspace(800, 400, 1024, endpoint=False)

    # Form to have the correct shape for the dedisperse function to operate
    freq = np.zeros((1024, 2))
    freq[:, 0] = frequency
    freq[:, 1] = frequency

    binning = tab.meta['I_BIN']
    mjd = Time(tab.meta['T_START'], format='isot', precision=9).mjd
    #tab = QTable.read(f'{istream}GP_tab-mjd_{mjd:.2f}-sigma_{sigma}-binning_{binning}.fits')

    tbin = tab.meta['TBIN'] * u.s
    dm = dm * u.pc / u.cm**3 # Set up the dispersion measure.

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

    T_START = time.time()

    tab.sort('snr')
    tab = tab[::-1] #tab.reverse()
    #print(tab)

    prepulse = 2625
    pulse_width = 15625

    POS = tab['pos']

    FNAMES = []

    START_TIMES = []

    id = 0

    t0 = time.time()

    pulse_limit = 512

    for pos in [POS[0]]:

        fname = 'pulse_{:04d}.npy'.format(id)
        FNAMES += [fname]

        start_time0 = out_frames[0].start_time

        start_times = []
        i = 0

        pulse = np.zeros((pulse_width, 1024, 4), dtype=np.float16)#2), dtype=np.complex64)

        for frame in out_frames:

            pulse_time = (pos * binning - prepulse) * tbin
            dstart = (frame.start_time - start_time0).to(u.s)
            dtau = frame.dm.time_delay(frame.frequency, freq[0,0]*u.MHz)
            corr = dtau[0] - dstart

            for _ in range(8):
                chantime = frame.start_time + (pos * binning - prepulse) * tbin
                start_times += [chantime.isot]
            try:    
                frame.seek(pulse_time - dtau[0] + corr)
                readpulse = frame.read(pulse_width)        

                #pulse[:,i:i+8,:] = frame.read(pulse_width)
                pulse[:,i:i+8,0] = np.real(readpulse[:,:,0])
                pulse[:,i:i+8,1] = np.imag(readpulse[:,:,0])
                pulse[:,i:i+8,2] = np.real(readpulse[:,:,1])
                pulse[:,i:i+8,3] = np.imag(readpulse[:,:,1])        

                #print('Channels {}-{} loaded: {}% Complete'.format(i, i+8, 100*(i+8)/1024), end='                 \r')
            except:
                pass
            i += 8

    I = np.sum(pulse**2, axis=2).astype(np.float64)
    
    dm, s = dm_fit(I, dm, Niter=4, L=50)
    best_dm = dm[np.argmax(s)]
    dm = dm[np.argmax(s)].value

tab.meta['DM_guess'] = dm


tab.write(istream + f'search_tab.fits', overwrite=True)
