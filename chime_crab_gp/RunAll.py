import os

homedir = '/home/serafinnadeau/'
datadir = '/drives/CHA/'
codedir = homedir + 'Scripts/Chime_Crab_GP/chime_crab_gp/'

#testdir = '/drives/CHF/4/crab_gp_tests/'
testdir = '/pulsar-baseband-archiver/crab_gp_archive/'

datestr = '20190721'
timestr = '173620'

os.chdir(testdir)
os.system(f'mkdir {datestr}')
os.chdir(datestr)
os.system('mkdir splitdir')
os.system('mkdir istream')
os.system('mkdir pulsedir')

os.chdir(codedir)

os.system(f'python3 ChannelSplit.py {datestr} {timestr}')

os.chdir(codedir)

os.system(f'python3 StreamSearch.py {datestr} {timestr}')

os.chdir(codedir)

os.system(f'python3 ChannelDedisperse.py {datestr} {timestr}')


os.chdir(f'{testdir}{datestr}')
os.system('rm -r splitdir')

