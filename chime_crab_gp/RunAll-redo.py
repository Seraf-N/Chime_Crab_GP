#!/usr/bin/env python3

from datetime import datetime
from astropy.time import Time
from astropy.table import Table, QTable
import astropy.units as u
import numpy as np
from time import time
from time import sleep

import os
import sys
if '/home/serafinnadeau/Python/packages/scintillometry/' not in sys.path:
    sys.path.append('/home/serafinnadeau/Python/packages/scintillometry/')

homedir = '/home/serafinnadeau/'
datadir = '/drives/CHF/'
#datadir = '/drives/CHF/'
codedir = homedir + 'Scripts/Chime_Crab_GP/chime_crab_gp/'

testdir = '/drives/CHF/8/crab_archive/'
plots = testdir + 'Plots/'
#testdir = '/pulsar-baseband-archiver/crab_gp_archive/'


datestr = '20200312'
timestr = '021310'
#20200311T021712

print(datestr, timestr)


os.system(f'/bin/mkdir {testdir}')
os.system(f'/bin/mkdir {testdir}Plots')
os.system(f'/bin/mkdir {testdir}{datestr}')
os.system(f'/bin/mkdir {testdir}{datestr}/splitdir')
os.system(f'/bin/mkdir {testdir}{datestr}/istream')
os.system(f'/bin/mkdir {testdir}{datestr}/pulsedir')

os.system(f'/home/serafinnadeau/Python/anaconda3/bin/python3 {codedir}ChannelSplit.py {datestr} {timestr}')

os.system(f'/home/serafinnadeau/Python/anaconda3/bin/python3 {codedir}StreamSearch.py {datestr} {timestr}')
os.system(f'/home/serafinnadeau/Python/anaconda3/bin/python3 {codedir}StreamSearch.py {datestr} {timestr}')

os.system(f'/home/serafinnadeau/Python/anaconda3/bin/python3 {codedir}ChannelDedisperse.py {datestr} {timestr}')

os.chdir(codedir)
from Analysis_tools import *
splitdir = testdir + f'{datestr}/splitdir/'
istream = testdir + f'{datestr}/istream/'
pulsedir = testdir + f'{datestr}/pulsedir/'

try:
    ptab = QTable.read(istream + 'pulse_tab.fits')
    os.system(f'/bin/rm -r {testdir}{datestr}/splitdir')
except (OSError, FileNotFoundError):
    print('Pulsetab not written. Splitdir not deleted')

ptab = QTable.read(istream + 'pulse_tab.fits')
stab = QTable.read(istream + 'search_tab.fits')

fig = plt.figure('end_plots', figsize=(20, 10))

plt.subplot(244)
nu, nu_err = crab_nu_fit(stab, N=100)
nu, nu_err = crab_nu_fit(stab, N=100, start=nu-nu_err, end=nu+nu_err)
plt.scatter(stab['off_s'][::-1].value%(1/nu) * nu, stab['off_s'][::-1])
plt.scatter(ptab['off_s'][::-1].value%(1/nu) * nu, ptab['off_s'][::-1])
plt.ylabel('Time [s]')
plt.grid()

maxT = np.max(ptab['off_s'])
m = ptab['off_s'] < maxT - 1*u.s
tab = ptab[m]
m = tab['off_s'] > 5*u.s
tab = tab[m]
i = int(tab['fname'][0][6:10])

pulse = read_pulse(datestr, i)
I = np.sum(pulse**2, axis=2).astype(np.float64)
no = i*1

plt.subplot(248)
plt.scatter(stab['off_s'][::-1].value%(1/nu) * nu, stab['snr'][::-1], label=fr'$\nu$={nu:.4f}Hz')
plt.scatter(ptab['off_s'][::-1].value%(1/nu) * nu, ptab['snr'][::-1])
plt.grid()
plt.ylabel('Peak S/N')
plt.xlabel('Phase')
plt.yscale('log')
plt.legend(loc=0)

plt.subplot(243)
dm, s = dm_fit(I, stab.meta['DM_GUESS'], Niter=4, L=200)
best_dm = dm[np.argmax(s)]
DM0 = DispersionMeasure(stab.meta['DM_GUESS'])
plt.plot(np.array(DispersionMeasure(dm)), s, '.', label=f'DM = {best_dm.value:.4f} pc/cm^3')
plt.xlim(best_dm.value-0.005, best_dm.value+0.005)
plt.xlabel('DM [pc/cm^3]')
plt.ylabel('Peak S/N')
plt.legend(loc=0)
plt.grid()

plt.subplot(241)
phase = np.arange(0, 15625, 1) * 2.56e-6 * nu
I_shift = dedisperse_chunk(I*1, best_dm - DM0)
i_shift = np.sum(I_shift, axis=1)
plt.plot(phase, i_shift)
pos = np.argmax(i_shift) * 2.56e-6 * nu
plt.xlim(pos-0.01, pos+0.02)
plt.xlabel('Phase')
plt.ylabel('Power [arb.]')

plt.subplot(245)
freq = np.linspace(800, 400, 1024, endpoint=False)
plt.imshow(I.transpose(), aspect='auto', extent=[phase[0], phase[-1], freq[-1], freq[0]], vmin=0, vmax=10)
plt.ylabel('Frequency [MHz]')
plt.xlabel('Pulse phase [cycle]')
plt.xlim(pos-0.01, pos+0.04)

plt.subplot(247)
f = np.median(I, axis=0)
m = f == 0
f[m] = np.nan
plt.plot(f, freq, label='meadian', alpha=0.3)
s = np.std(I, axis=0)
m = s == 0
s[m] = np.nan
plt.plot(s, freq, label='standard dev.', alpha=0.3)
plt.legend(loc=0)

i = int(tab['fname'][1][6:10])
pulse = read_pulse(datestr, i)
I = np.sum(pulse**2, axis=2).astype(np.float64)
no = i*1

plt.subplot(242)
phase = np.arange(0, 15625, 1) * 2.56e-6 * nu
I_shift = dedisperse_chunk(I*1, best_dm - DM0)
i_shift = np.sum(I_shift, axis=1)
plt.plot(phase, i_shift)
pos = np.argmax(i_shift) * 2.56e-6 * nu
plt.xlim(pos-0.01, pos+0.02)

plt.subplot(246)
freq = np.linspace(800, 400, 1024, endpoint=False)
plt.imshow(I.transpose(), aspect='auto', extent=[phase[0], phase[-1], freq[-1], freq[0]], vmin=0, vmax=10)
plt.xlim(pos-0.01, pos+0.04)
plt.xlabel('Pulse phase [cycles]')

fig.suptitle(f'{datestr} Summary')

plt.savefig(istream + f'end_plot.png')
plt.savefig(plots + f'{datestr}_end_plot.png')

os.system(f'/bin/echo Summary plot for {datestr} | /bin/mailx -r nadeau@cita.utoronto.ca -s "{datestr} Summary plot" -a {plots}{datestr}_end_plot.png serafinnadeau@astro.utoronto.ca')
print('Email sent')
