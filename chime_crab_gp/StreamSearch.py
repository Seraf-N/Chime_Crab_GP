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

w = 31250 # Chunk width
dm = 56.7 # Stock DM for Crab

stream1D = master(I, w=w, dm=dm, s_per_sample=s_per_sample)

#########################################################################################
# Correct 1D time stream for variations in background levels
#########################################################################################

stream1D = correct_stream(stream1D, N=300)

print('Corrections applied')

#############################################################################
# Search intensity stream for Giant Pulses
#############################################################################

tab = streamsearch(stream1D, splittab, 3.0, banddir, datestr, timestr, 
                   Nmax=1000, dm=DispersionMeasure(dm))


