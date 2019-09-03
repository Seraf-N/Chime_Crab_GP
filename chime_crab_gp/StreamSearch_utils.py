import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.table import QTable

from numpy.lib.format import open_memmap
import os

import time

from utils import DispersionMeasure, imshift

def master(stream2D, w=31250, dm=56.7, s_per_sample=2.56e-4, verbose=True):
    '''
    Takes a memmap 2D stream and converts it to a 1D dedispersed intensity 
    stream.
    It does this chunk by chunk, with the time width of each chunk defined by
    the input parameter w and dedisperses to the input dm parameter.
    '''

    # Obtain data for w samples at a time, accounting for lost samples due to 
    # dedispersion.
    dm = DispersionMeasure(dm)
    dt = dm.time_delay(800*u.MHz, 400*u.MHz)

    time_waste = int(abs(dt.value / s_per_sample) + 1)
    print(f'{time_waste} samples lost at the end of array due to dedispersion')

    w_eff = w - time_waste # the effective width of each read chunk after dedispersion

    N = int(len(stream2D) / w_eff) # the number chunks to be read in

    stream1D = np.zeros(N * w_eff)

    if verbose:
        t0 = time.time()
        chunk_n = -1
        verbose_print(chunk_n, N, t0, extra='')

    for chunk_n in range(N):		

        sample_min = select_chunk(chunk_n, w_eff) # Calculate starting time bin of chunk

        if verbose:
            verbose_print(chunk_n, N, t0, extra='Reading')
        chunk = read_chunk(stream2D, w, sample_min) # Read in chunk

        if verbose:
            verbose_print(chunk_n, N, t0, extra='Masking')
        chunk = mask_chunk(chunk)

        if verbose:
            verbose_print(chunk_n, N, t0, extra='Dedispersing')
        chunk = dedisperse_chunk(chunk, dm, s_per_sample)

        if verbose:
            verbose_print(chunk_n, N, t0, extra='Adding')
        stream1D = fuse_chunk(stream1D, chunk, sample_min, w_eff)

    if verbose:
        verbose_print(chunk_n, N, t0, extra='Complete', rewrite=False)
        print('')

    return stream1D

def verbose_print(current, max, t0, extra='', rewrite=True):
    tf = time.time()
    ss = int(tf-t0)
    hh = ss // 3600
    mm = ss // 60 - hh*60
    ss = ss % 60
    if rewrite:
        end = '                          \r'
    else:
        end = '                          \n'
    print(f'{current+1:0{len(str(max))}}/{max}: '
          f'{(current+1)*100/max:05.2f}% complete'
          f' -- {hh:02d}:{mm:02d}:{ss:02d} elapsed -- {extra: <20} ',
          end=end)
    return

def select_chunk(chunk_n, w_eff):
    '''
    Calculates the starting sample of the memmaped 2D stream for a given chunk
    number.
    '''
    sample_min = w_eff * chunk_n
    return sample_min

def read_chunk(stream2D, w, sample_min):
    '''
    Reads in a time chunk of width w time bins, starting at sample_min from a 
    2D memmaped stream.
    Reshapes to (FREQ, TIME) / (1024, w)
    '''
    #stream2D.seek(sample_min)
    #chunk = stream2D.read(w)
    chunk = stream2D[sample_min:sample_min+w] * 1
    shape = np.shape(chunk)
    if shape[1] == 1024:
        chunk = chunk.transpose(1,0)

    return chunk

def mask_chunk(chunk):
    '''
    Replaces zero masking of RFI with the mean of the relevant frequency channel
    '''
    for i in range(len(chunk)): # Loop over frequencies
        row2 = chunk[i] * 1
        m = chunk[i] == 0
        if np.sum(m) != 0:
            row2[m] = np.nan            # Mask all true zeros to nan
            mean= np.nanmean(row2)      # Compute the mean of each channel
            if np.isnan(mean):
                chunk[i][m] = 0 # if channel mean is nan, the fill channel back with 0
            else:
                chunk[i][m] = mean # Fill gaps in channel with the channel mean value
        else:
            chunk[i] = chunk[i]
    return chunk

def dedisperse_chunk(chunk, dm, s_per_sample):
    '''
    Dedisperses the chunk with the given dm
    '''
    freqs = np.linspace(800, 400, 1024) * u.MHz
    dt = dm.time_delay(800*u.MHz, freqs)

    chunk = imshift(chunk*1, shiftc=dt.value/s_per_sample)

    return chunk

def fuse_chunk(stream1D, chunk, sample_min, w_eff):
    '''
    Collapses chunk and adds it to the dedispersed 1D stream
    '''
    stream = np.sum(chunk, axis=0)
    stream1D[sample_min:sample_min+w_eff] = stream[:w_eff]

    return stream1D

def correct_stream(stream1D, N=300):
    '''
    flattens the background levels of the intensity stream to better pick out 
    giant pulses in the search by itteratively subtracting the rolling mean.
    Saves the corrected 1D stream as a memmap object.
    '''
    mean_test = np.nanmean(stream1D[:-5000])
    std_test = np.nanstd(stream1D[:-5000])
    snr_test = (stream1D[:-5000]-mean_test)/std_test

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

    stream1D = open_memmap('istream_corr.npy', dtype=np.float32, mode='w+', 
                           shape=np.shape(snr_test_4))
    stream1D[:] = snr_test_4
    
    return stream1D

def streamsearch(stream1D, splittab, cutoff, banddir, datestr, timestr, 
                 Nmax=1024, dm=DispersionMeasure(56.7), output=False):
    '''
    Searches the corrected 1D stream for signals stonger than 'cutoff' sigma
    '''
    POS = []
    SNR = []
    snr_search = stream1D * 1

    start_time = Time(splittab.meta['T_START'], format='isot', precision=9)
    n_frames = splittab.meta['NFRAMES']
    samples_per_frame = splittab.meta['FRAMELEN']
    binning = splittab.meta['I_BIN']
    s_per_sample = splittab.meta['TBIN'] * binning
    
    nsamples = n_frames * samples_per_frame

    pos = np.argmax(snr_search)
    snr = snr_search[pos]

    i = 0
    t0 = time.time()

    snr_search[:int(1.11/2.56e-6/100)] = 0

    while snr > cutoff:
        if len(POS) < Nmax:
            # Ensure that the pulse is not too close to start of data for dedispersion
            if pos * 100 * 2.56e-6 > 1.11:
                POS += [pos]
                SNR += [snr]

            snr_search[pos-100:pos+100] = 0

            pos = np.argmax(snr_search)
            snr = snr_search[pos]

            i += 1
            t = time.time() - t0
            m, s = divmod(t, 60)
            h, m = divmod(m, 60)
            print(f'Intensity stream searched for pulses: {len(POS)} pulses found -- '
                  f'S/N: {snr:.3f} -- POS: {pos*100*2.56e-6:.3f} -- Time elapsed: '
                  f'{int(h):02d}:{int(m):02d}:{int(s):02d}', end='                     \r')
            if len(POS) == 1000:
                break

    print(f'Intenisty stream searched for pulses: {len(POS)} pulses found                             ')

    POS = np.array(POS)
    TIME_S = POS * s_per_sample
    SNR = np.array(SNR)
    MJD = start_time + TIME_S * u.s
    
    # Create Table of GPs to be saved

    tab = QTable()
    tab.meta = splittab.meta

    tab['time'] = (TIME_S * u.s + start_time).isot

    tab['off_s'] = TIME_S * u.s
    tab['pos'] = POS
    tab['snr'] = SNR
    tab.sort('pos')


    tab.meta['DM'] = dm.value
    tab.meta['binning'] = 100
    tab.meta['sigma'] = cutoff
    tab.meta['start'] = start_time.isot
    tab.meta['nsamples'] = nsamples
    tab.meta['history'] = ['Intensity stream i_stream.npy saved from ChannelSplit'
                           f' on vdif files {banddir}*/{datestr}T{timestr}'
                           'Z_chime_psr_vdif/*',
                           'i_stream.npy dedispersed and searched for giant pulses']

    tab.write(f'search_tab.fits', overwrite=True)

    if output:
        return tab
    return
