#!/usr/bin/env python3

from datetime import datetime
from astropy.time import Time
from astropy.table import Table
import numpy as np
from time import time
from time import sleep

import os

homedir = '/home/serafinnadeau/'
datadir = '/drives/CHA/'
#datadir = '/drives/CHF/'
codedir = homedir + 'Scripts/Chime_Crab_GP/chime_crab_gp/'

#testdir = '/drives/CHF/4/crab_gp_tests/'
testdir = '/pulsar-baseband-archiver/crab_gp_archive/'


get_time = True


if get_time == False:
    datestr = '20190901'
    time = Time(datestr[:4]+'-'+datestr[4:6]+'-'+datestr[6:]+'T00:00:00')
    
    
else:
    time = Time(datetime.now())

schedulefile = f'/home/cng/CHIME-PSR-ScheduleFiles/Schedule_JD{int(time.jd-0.5)}.5.dat.bb'
txt = open(schedulefile, 'r')

lines = txt.readlines()
for line in lines:
    if 'B0531+21' == line[:8]:
        row = line

txt.close()

row = row.split()
daytab = Table(np.array(row), 
               names=['psr', 'unix_start', 'unix_end', 'ra', 'dec', 'beam_no', 
                          'scaling_factor', 'dm', 'mode', 'phase_bin', 'raj2000', 'decj2000'], 
               dtype=[str, np.float64, np.float64, np.float64, np.float64, 
                      np.int16, np.float16, np.float32, str, np.int16, np.float32, np.float32])

bbstart = datetime.fromtimestamp(daytab['unix_start'])
bbend = datetime.fromtimestamp(daytab['unix_end'])

if get_time:
    dt = datetime.now() - bbend
    while dt.total_seconds()/60 < 10:
        sleep(10 - dt.total_seconds()/60)    
        dt = datetime.now() - bbend

vdifdirs = os.listdir('/drives/CHA/0')
for vdifdir in vdifdirs:
    
    if vdifdir[:8] == f'{bbstart.year:04d}{bbstart.month:02d}{bbstart.day:02d}':
        vdif_time = vdifdir[9:15]
        hh = int(vdif_time[:2])
        mm = int(vdif_time[2:4])
        ss = int(vdif_time[4:])
        vdif_time_s = hh * 3600 + mm * 60 + ss
        hh = bbstart.hour
        mm = bbstart.minute
        ss = bbstart.second
        unix_time_s = hh * 3600 + mm * 60 + ss
        if abs(vdif_time_s - unix_time_s) % (24*3600) <= 30: 
            print(vdifdir)
            datestr = vdifdir[:8]
            timestr = vdifdir[9:15]

os.chdir(testdir)
os.system(f'mkdir {datestr}')
os.chdir(datestr)
os.system('mkdir splitdir')
os.system('mkdir istream')
os.system('mkdir pulsedir')

os.chdir(codedir)

os.system(f'nice 19 python3 ChannelSplit.py {datestr} {timestr}')

os.chdir(codedir)

os.system(f'nice 19 python3 StreamSearch.py {datestr} {timestr}')

os.chdir(codedir)

os.system(f'nice 19 python3 ChannelDedisperse.py {datestr} {timestr}')

os.chdir(f'{testdir}{datestr}')
os.system('rm -r splitdir')
os.chdir(codedir)
