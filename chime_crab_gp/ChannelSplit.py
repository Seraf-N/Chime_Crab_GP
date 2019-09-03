#import sys
#sys.path.append('/home/serafinnadeau/Python/packages/scintillometry')

import numpy as np

import os
import baseband
from baseband import vdif

import astropy.units as u
from astropy.table import QTable

from scintillometry.shaping import ChangeSampleShape
from scintillometry.combining import Concatenate

from scipy.stats import binned_statistic as bs

import time
from functools import reduce

from numpy.lib.format import open_memmap

from utils import imbin




'''# CITA DIRECTORIES WORKSPACE
workdir = '/mnt/scratch-lustre/nadeau/Chime/'
codedir = '/mnt/scratch-lustre/nadeau/Chime/Code'
banddir = '/mnt/scratch-lustre/hhlin/Data/20181019T113802Z_chime_psr_vdif'
plotdir = '/mnt/scratch-lustre/nadeau/Chime/Code/Plots'
tempdir = '/mnt/scratch-lustre/nadeau/Chime/Data/20181019T113802Z_temp_vdif'

splitdir= '/mnt/scratch-lustre/nadeau/Chime/Data/20181019T113802Z_split_vdif'
istream = '/mnt/scratch-lustre/nadeau/Chime/Data/20181019T113802Z_i_stream'

os.chdir(banddir)

# Retrieve list of vdif files
x = os.listdir()
x.sort()

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
#banddir = '/drives/CHF/'

testdir = '/pulsar-baseband-archiver/crab_gp_archive/' 
splitdir = testdir + f'{datestr}/splitdir/'
istream = testdir + f'{datestr}/istream/'

os.chdir(banddir)

# Retrieve list of vdif files
ftab = QTable()
fnames = []
fnumbers = []

for i in range(8):
    X = os.listdir(f'{i}/{datestr}T{timestr}Z_chime_psr_vdif/')
    for x in X:
        fnames += [f'{i}/{datestr}T{timestr}Z_chime_psr_vdif/' + x]
    
for f in fnames:
    fnumbers += [f[34:41]]

ftab['fname'] = fnames
ftab['fnumber'] = fnumbers

ftab.sort('fnumber')
x = list(ftab['fname'][:-8])
#x = x[:2000]

# Open dataset
os.chdir(banddir)

data = vdif.open(x, 'rs', sample_rate=1/(2.56*u.us), verify=False)

# Reshape from CHIME format to a sensible format (time, freq, pol) = (time, 1024, 2)     

def reshape(data):
    pols_sep = data.reshape(-1, 256, 4, 2)
    freq_t = pols_sep.transpose(0, 2, 1, 3)
    return freq_t.reshape(-1, 1024, 2)

shaped = ChangeSampleShape(data, reshape)

# Read in data in chunks

datatime = len(x) * 3125 * 2.56e-6 # no of files x samples per file x s per sample -> time in s of total dataset
samples_per_frame = 15625 # This gives 0.04s of data / frame
frames_per_second = 25
n_frames = int(frames_per_second * (datatime))#-1)) # Total number of frames with samples_per_frame sample of 2.56us each. 25 frames / s, with frame length defined as above
start_time = data.start_time

# Add metadata to table for following parts of pipeline to use as reference
ftab.meta['DATATIME'] = datatime
ftab.meta['FRAMELEN'] = samples_per_frame
ftab.meta['NFRAMES'] = n_frames
ftab.meta['T_START'] = start_time.isot
ftab.meta['T_STOP'] = data.stop_time.isot
ftab.meta['TBIN'] = (1 / data.sample_rate).to('s').value

binning = 100
ftab.meta['I_BIN'] = binning

os.chdir(istream)
ftab.write('SplitTab.fits', overwrite=True)
os.chdir(banddir)

t0 = time.time()

##########################################################################
# Open vdif files for each set of 8 channels to write data to
##########################################################################

os.chdir(splitdir)  
out_frames = []
for i in range(0, 1024, 8):
    
    fname = 'Split_Channel_c{:04d}-{:04d}.vdif'.format(i, i+7)
    chanframe = vdif.open(fname, 'ws', sample_rate=1/(2.56*u.us), 
                          samples_per_frame=samples_per_frame, 
                          time=start_time, nchan=8, nthread=2, 
                          complex_data=True, edv=0, station='CX', 
                          bps=4)
    out_frames += [chanframe] 

print(f'VDIF files opened')

###########################################################################
# Set up memmap for saving binned intensity stream
###########################################################################

intensities = [] # array to store 4 frames unbinned 
i_frame = 625
os.chdir(istream)
i_stream = open_memmap('i_stream.npy', dtype=np.float32, 
                       mode='w+', shape=(int(samples_per_frame*n_frames/binning), 
                       1024))
i_start = 0
i_end = i_start + i_frame

print(f'Intensity stream memmap initialized')

##########################################################################
# Loop through frames of data and split into channels
# +
# Save intensity stream
##########################################################################

os.chdir(banddir)
start_frame = 0#n_frames - 30
#n_frames = int(1  * n_frames)    
shaped.seek(start_frame * samples_per_frame)
for frame in range(start_frame, n_frames):
    
    # Read data frame
    os.chdir(banddir)    
    dataframe = shaped.read(samples_per_frame)
    
    frame_start = start_time + 2.56*u.us*frame
    
    # Compute intensity frame and append to array
    intensities += [np.absolute(dataframe[:,:,0])**2 + np.absolute(dataframe[:,:,1])**2]
    
    # Once sufficient number of frames loaded, bin intensity stream by 100 and save to memmap
    if len(intensities) == 4:
        
        I = np.zeros((i_frame * binning, 1024))
        count = 0
        
        for i in intensities:
            I[samples_per_frame*count:samples_per_frame*count+samples_per_frame] = i
            count += 1
            
        del intensities
        intensities = []
            
        I_bin, _, _, _ = imbin(I, binning_r=binning)
        del I
        
        os.chdir(istream)
        i_stream[i_start:i_end] = I_bin
        i_start += i_frame
        i_end = i_start + i_frame
        del I_bin
        
        os.chdir(banddir)  
    
    # Loop over sets of 8 frequency channels and write frames for each one
    for chan, chanframe in zip(range(0, 1024, 8), out_frames):
        
        framechannel = dataframe[:,chan:chan+8,:]
        chanframe.write(framechannel.transpose(0, 2, 1))
        del framechannel
        
    del dataframe
    t = time.time() - t0
    hh = int(t/3600)
    mm = int((t % 3600)/60)
    ss = int(t % 60)
    print(f' Splitting fileno {frame+1 - start_frame}/{n_frames-start_frame}: {(frame+1-start_frame)*binning/(n_frames-start_frame):.4f}% complete; Time elapsed = {hh:02d}:{mm:02d}:{ss:02d}', end='            \r')

for chanframe in out_frames:
    chanframe.close()

print(' ')
print('\n Channel VDIF files closed')

os.chdir(istream)
np.savetxt('complete.txt', [hh, mm, ss])
