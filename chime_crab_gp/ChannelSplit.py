import numpy as np

import os
import baseband
from baseband import vdif

import astropy.units as u

from scintillometry.shaping import ChangeSampleShape
from scintillometry.combining import Concatenate

from scipy.stats import binned_statistic as bs

import time
from functools import reduce

workdir = '/mnt/scratch-lustre/nadeau/Chime/'
codedir = '/mnt/scratch-lustre/nadeau/Chime/Code'
banddir = '/mnt/scratch-lustre/hhlin/Data/20181019T113802Z_chime_psr_vdif'
plotdir = '/mnt/scratch-lustre/nadeau/Chime/Code/Plots'
tempdir = '/mnt/scratch-lustre/nadeau/Chime/Data/20181019T113802Z_temp_vdif'

splitdir= '/mnt/scratch-lustre/nadeau/Chime/Data/20181019T113802Z_split_vdif'
istream = '/mnt/scratch-lustre/nadeau/Chime/Data/20181019T113802Z_i_stream'

os.chdir(banddir)

def reshape(data):
    '''
    Function to reshape CHIME Baseband data to (time, ferquency, polarization) 
    shape of (-1, 1024, 2)
    '''
    pols_sep = data.reshape(-1, 256, 4, 2)
    freq_t = pols_sep.transpose(0, 2, 1, 3)
    return freq_t.reshape(-1, 1024, 2)

from utils import imbin

# Retrieve list of vdif files
x = os.listdir()
x.sort()

# Open dataset
os.chdir(banddir)

data = vdif.open(x, 'rs', sample_rate=1/(2.56*u.us), verify=False)

# Read in data in chunks

samples_per_frame=15625 # This gives 0.04s of data / frame
n_frames = 25*320 # Total number of frames with samples_per_frame sample of 2.56us each. 25 frames / s, with frame length defined as above
start_time = data.start_time

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

###########################################################################
# Set up memmap for saving binned intensity stream
###########################################################################

intensities = [] # array to store 4 frames unbinned 
i_frame = 625
os.chdir(istream)
i_stream = np.memmap('i_stream.dat', dtype=np.float32, mode='w+', shape=(int(samples_per_frame*n_frames/100), 1024))
i_start = 0
i_end = i_start + i_frame

##########################################################################
# Loop through frames of data and split into channels
# +
# Save intensity stream
##########################################################################

os.chdir(banddir)    
shaped.seek(0)
for frame in range(n_frames):
    
    # Read data frame
    os.chdir(banddir)    
    dataframe = shaped.read(samples_per_frame)
    
    frame_start = start_time + 2.56*u.us*frame
    
    # Compute intensity frame and append to array
    intensities += [np.absolute(dataframe[:,:,0])**2 + np.absolute(dataframe[:,:,1])**2]
    
    # Once sufficient number of frames loaded, bin intensity stream by 100 and save to memmap
    if len(intensities) == 4:
        
        I = np.zeros((62500, 1024))
        count = 0
        
        for i in intensities:
            I[samples_per_frame*count:samples_per_frame*count+samples_per_frame] = i
            count += 1
            
        del intensities
        intensities = []
            
        I_bin, _, _, _ = imbin(I, binning_r=100)
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
    print('Splitting fileno {}/{}: {}% complete; '.format(frame+1, n_frames, (frame+1)*100/n_frames), 'Time elapsed = {}:{}:{}'.format(hh, mm, ss), end='             \r')

for chanframe in out_frames:
    chanframe.close()

print('Channel VDIF files closed')

os.chdir(istream)
np.savetxt('complete.txt', [hh, mm, ss])