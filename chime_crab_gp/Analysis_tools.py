import numpy as np
from astropy.table import QTable
import astropy.units as u

def crab_nu_fit(ptab, start=26., end=26.3, N=1000, plot=False):
    '''
    From list of GP detections, calculates the pulse period by wrapping
    the detection times by the 1/(pulse frequency) and estimating the 
    area under the curve obtained from the cumulative index as a function of 
    phase (corresponding to nu).

    
    '''

    ptab = ptab*1
    ptab.sort('off_s')
    time = ptab['off_s']
    snr = ptab['snr']

    I = []
    nus = np.linspace(start, end, N)

    i = 1
    for nu in nus:
        # Wrap the detection time uising the guess frequency to get a phase
        phase = (time.value / (1 / nu)) % 1
        # Get the cumulative number of detections as a function of phase
        count_sum = np.cumsum(range(len(phase)))
        # Estimate the area under coun_sum using trapezoidal method
        I += [np.trapz(count_sum, x=phase)]
        
        if plot:
            if i == 0:
                plt.figure('Phase')
                plt.xlabel('Phase [cycle]')
                plt.ylabel('Detection S/N')
                plt.figure('Cumulative Index')
                plt.xlabel('Phase [cycle]')
                plt.ylabel('Cumulative number of detections')
            if (i*100/N)%10 == 0:
                plt.figure('Phase')
                plt.plot(phase, snr, '.', alpha=0.3)
                plt.figure('Cumulative Index')
                plt.plot(phase, count_sum, alpha=0.3, label=f'{nu:.4f}')
            i += 1
    
    # Fit I for the pulsar frequency
       
    if plot:
        plt.figure('Integral')
        plt.xlable(r'$\nu$ [Hz]')
        plt.ylabel('Integral of cumulative number of detections vs. phase')
        plt.plot(nus, I)
            
    ########################################################################
    # Return estimate of nu. Needs to be made more rigorious by fitting for 
    # it in future, with error above and below estimate. For now, good guess.
    ########################################################################
    return nus[np.argmax[I]]


