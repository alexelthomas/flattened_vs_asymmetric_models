import numpy as np
import matplotlib.pyplot as plt
#import scipy.optimize as op

#likelihood function
class lnlike_mcmc(object):

    def __init__(self, _freq, _power, _model):
        self.freq = _freq
        self.power = _power
        self.model = _model

    def __call__(self, params):
        """
        Construct likelihood function for chi^2 2 d.o.f as given in Anderson (1990).

        Inputs: 
            :type freq: array-like
            :param freq: Array containing array of frequency values in fitting region

            :type power: array-like
            :param power: Array containing array of power values in fitting region.

            :type params: array-like
            :param params: Array containing model parameters

        Output:
            :type like: float
            :param like: Negative Log-likelihood value for data given input parameters
        """
        # Construct model for given set of parameters
        mod = self.model(params)

        # Input into equation (11) from Anderson (1990)
        # But we want log-likelihood not negative log-likelihood (in MCMC)
        # and so we add the -1.0
        like = np.sum(np.log(mod) + (self.power / mod))
        return -1.0*like

#prior
class lnprior(object):
    
    def __init__(self, _bounds):
        self.bounds = _bounds
        
    def __call__(self, params):
        # A fancy way of ensuring all parameters lie within the given bounds
        if not all(b[0] < v < b[1] for v, b in zip(params, self.bounds)):
            return -np.inf
        lnprior = 0
        lnprior += np.log(np.sin(np.radians(params[-2])))
        return lnprior

#probability distribution
class lnprob(object):
    
    def __init__(self, _like, _prior):
        self.like = _like
        self.prior = _prior
        
    def __call__(self, params):
        lp = self.prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.like(params)
    
#model
class Model(object):
    
    def __init__(self, _freq):
        self.freq = _freq
        
    def lorentzian(self, params):
        """
        Compute Lorentzian profile given input parameters
        Inputs: 
            :type freq: array-like
            :param freq: Array containing array of frequency values

            :type params: array-like
            :param params: Array containing model parameters

        Output:
            :type m: array-like
            :param m: Lorentzian profile created given input parameters
        """
        height, width, frequency = params

        return height / (1.0+ (4.0 / width**2)*(self.freq - frequency)**2)
    
    def height(self,params):
        """
        Calculate height of mode
        Inputs:
            :type params: array-like
            :param params: Array containing model parameters for amplitude and height of mode
            
        Output:
            :type height: array-like
            :param height: Height of mode 
        """
        amplitude, width = params
        return 2.0 * amplitude **2 / (np.pi * width)
    
    def X_value(self,params):
        """
        Calculate value of X
        Inputs:
            :type freq: array-like
            :param freq: Array containing array of frequency values

            :type params: array-like
            :param params: Array containing model parameters for central frequency, width and splitting of mode.
            
        Output:
            :type X: array-like
            :param X: X value needed in model using Toutain 2008
        """
        frequency, width, splitting = params
        return (self.freq-(frequency+0.5*splitting))/(0.5*width)
    
    def __call__(self, params):
        """
        Flattened Lorentzian as described in Toutain et al., 2008.
        
        Inputs: 
            :type freq: array-like
            :param freq: Array containing array of frequency values

            :type params: array-like
            :param params: Array containing model parameters

        Output:
            :type m: array-like
            :param m: model created given input parameters
        """
        # Parameters array is set out as follows
        amplitude0, amplitude1, amplitude2, frequency0, frequency1, frequency2, width0, width1, width2, splitting1, splitting2, inclination, background = params

        # Set up model array
        m = np.zeros(len(freq))
        
        epsilon1 = splitting1/width1
        epsilon2 = splitting2/width2
        
        # To get flat top rather than drop in middle of model we must make sure that arctan(*)>0
        # This model involves sum of l=0,1&2 modes which I do in steps
        #l=0 step
        m += self.lorentzian([self.height([amplitude0,width0]), width0,frequency0])
        #l=1 step
        m += self.height([amplitude1,width1])/(2.0*epsilon1) * np.arctan(np.abs(2.0*epsilon1/(1.-epsilon1**2+self.X_value([frequency1,width1,splitting1])**2)))
        #l=2 step
        m += self.height([amplitude2,width2])/(2.0*epsilon2) * np.arctan(np.abs(2.0*epsilon2/(1.-epsilon2**2+self.X_value([frequency2,width2,splitting2])**2)))
        
        return m + background #modelled flattened lorentzian + background

#generate fake data for trialling model      
def generate_data(freq, params, model):
    """
    Generate artificial mode profile with chi^2 2 d.o.f. noise
    
    Inputs: 
        :type freq: array-like
        :param freq: Array containing array of frequency values
        
        :type params: array-like
        :param params: Array containing model parameters

    Output:
        :type data: array-like
        :param data: data created given input parameters
    """
    # Generate limit spectrum
    limit_spectrum = model(params)
    
    # Create data with noise from equation (A1) of Anderson (1990)
    data = -limit_spectrum * np.log(np.random.rand(len(freq)))
    return data, limit_spectrum
 
def lorentzian_individual(freq, height, width, cent_freq):
        return height / (1.0+ (4.0 / width**2)*(freq - cent_freq)**2)
 
#=================================================================================
#Generate data 
# Create frequency array
# bin width for 4-years of Kepler in uHz
'''bw = 1.0 / (4.0 * 365.0 * 86400.0) * 1e6
freq = np.arange(2800.0, 3200.0, bw)

# Create model instance
model_inst = Model(freq)

# Parameters for artificial data
# amplitude0,amplitude1,amplitude2, frequency0,frequency1,frequency2, width0,width1,width2, splitting1,splitting2, inclination, background
trueparams = [1.0,1.0,1.0, 3010.0,3080.0,3000.0, 0.75,0.75,1.0, 0.75,1.0, 45.0, 0.2]

# Generate fake data and plot points as well as model used for generation
power, limit_model = generate_data(freq, trueparams, model_inst)
plt.figure('Generated data and true model')
plt.plot(freq, power, 'k')
plt.plot(freq, limit_model, 'r')
plt.xlabel(r'Frequency ($\mu$Hz)', fontsize=18)
plt.ylabel(r'PSD (arb. units)', fontsize=18)
'''    
 #load data ============================================================================
#data = np.genfromtxt('kplr008006161_kasoc-wpsd_slc_v1.pow')
data = np.genfromtxt('kplr008006161_kasoc-wpsd_slc_v1.pow.txt')
allfrequency = data[:,0]
allpower = data[:,1]

#Crop data to single l=1 multiplet
rangelength = 150.0
freqrangestart =  3480.0

freq_index= np.where(np.logical_and(allfrequency>freqrangestart , allfrequency<=freqrangestart+rangelength))[0]
frequency1 = allfrequency[freq_index[0]:freq_index[-1]]
power = allpower[freq_index[0]:freq_index[-1]]
bw = allfrequency[1]-allfrequency[0]
freq = np.arange(frequency1[0],frequency1[-1],bw)

if len(freq)!=len(power):
    freq = np.append(freq,freq[-1]+bw)

#frequency1 = allfrequency[minfreq:maxfreq-1]

# Create model instance
model_inst = Model(freq)
'''plt.figure('Data')
plt.plot(freq, power, 'k')
plt.xlabel(r'Frequency ($\mu$Hz)', fontsize=18)
plt.ylabel(r'PSD (arb. units)', fontsize=18)
'''

#==========================================================================================================================================
#FIND FIRST GUESSES
# Need to generate some first guesses

# Smooth data first, less susceptible to noise
import scipy.ndimage
# smooth with Gaussian kernel convolution
power_smooth = scipy.ndimage.filters.gaussian_filter1d(power,50.0)

# find all the peaks (get index number)
peaks = np.array([i for i in range(1, len(freq)-1)
                 if power_smooth[i-1] < power_smooth[i] and
                 power_smooth[i+1] < power_smooth[i]])

# For first guesses take the 3 largest peaks
# This gets the powers of the peaks, sorts them (by power) and chooses the indexes of the 3 highest powered peaks
idx = np.argsort(power_smooth[peaks])[-3:]
# Order chosen peaks by frequency (l=2,l=0,l=1) and then get the corresponding frequencies and power
# Use frequencies for estimates of central frequencies and power for amplitude estimates

idx_ordered = np.sort(freq[peaks][idx])
peak_freqs = [x for _,x in sorted(zip(idx_ordered,freq[peaks][idx]))]
peak_power = [x for _,x in sorted(zip(idx_ordered,power[peaks][idx]))]

# Estimate the splittings and widths as being the same for each peak
splitting_est = 1.0
width_est = 1.0

#Estimate inclination angle as intermediate
angle_est = 45.0

# Background can be worked out as taking average of bins in extrema of fitting region
numbins = 30
background_est = np.mean(np.r_[power[:numbins], power[-numbins:]])

# amplitude0,amplitude1,amplitude2, frequency0,frequency1,frequency2, width0,width1,width2, splitting1,splitting2, inclination, background
# First guess parameters
p0 = [peak_power[1],peak_power[2],peak_power[0], peak_freqs[1],peak_freqs[2],peak_freqs[0], width_est,width_est,width_est, 
      splitting_est,splitting_est, angle_est, background_est]
'''
plt.figure('First guess parameters')
plt.plot(freq, power, 'k')
plt.plot(freq, model_inst(p0), 'r') #model based on first guess parameters
plt.xlabel(r'Frequency ($\mu$Hz)', fontsize=18)
plt.ylabel(r'PSD (arb. units)', fontsize=18)
'''


#=======================================================================================
#==================================================================================
# SET UP PRIORS
# Create prior, likelihood instances
# For central frequency bounds we take the first guesses of central frequency and add a width where the bounds can exist
# amplitude0,amplitude1,amplitude2, frequency0,frequency1,frequency2, width0,width1,width2, splitting1,splitting2, inclination, background
freq_bounds_range = 4.0
bounds = [(0.0, 100.0),(0.0, 100.0),(0.0, 100.0), (peak_freqs[1]-freq_bounds_range,peak_freqs[1]+freq_bounds_range),(peak_freqs[2]-freq_bounds_range,peak_freqs[2]+freq_bounds_range),(peak_freqs[0]-freq_bounds_range,peak_freqs[0]+freq_bounds_range),
      (0.0,5.0),(0.0,5.0),(0.0,5.0), (0.0, 3.0),(0.0, 3.0), (0.0, 90.0), (0.0, 10.0)]
Prior = lnprior(bounds)
Lnlike = lnlike_mcmc(freq, power, model_inst)
Lnprob = lnprob(Lnlike, Prior)

# =================================================================================
#MCMC
# Define some labels
labels = [r'$A$', r'$\nu_{0}$', r'$\Gamma$', r'$\nu_{\mathrm{s}}$', r'$\nu_{\mathrm{a}}$',
          r'$i$', r'$B$']
#set up number of diensions (no. of parameters) and number of walkers conducting the random walk simultaneously
ndim, nwalkers = len(p0), 100
#distribute walkers using Gaussian with mean=0 and std=1
pos = np.array([p0 + 1e-4*np.random.randn(ndim) for i in range(nwalkers)])
print(np.shape(pos))
# Randomly scatter walkers according to isotropic distribution
pos[:,-2] = np.degrees(np.arccos(np.random.uniform(0, 1, len(pos))))
plt.hist(pos[:,-2], bins=10, normed=True, histtype='step')
plt.show()
import emcee
#conduct sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, Lnprob)#, threads=4)

# Clear and run the MCMC production chain with 1000 steps
print("Running MCMC...")
N_it = 1000.0
sampler.run_mcmc(pos, N_it, rstate0=np.random.get_state())
print("Done.")

#corner plot
import corner
print(np.shape(sampler.chain))
#discard first half of chain as burn in
samples = sampler.chain[:, int(0.8*N_it):, :].reshape((-1, ndim))
#corner.corner(samples, labels=labels)
#plt.show()

#chain/ sampler plot
#plt.plot(samples[:,0], alpha=0.5, color='k')


#==========================================================================================
#get parameters from MCMC. Involves taking median for the true value and the +/- 
#  error as the 68.3% credible interval (16th and 84th percentiles)
"""params = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
#[1.0, 5.0, 0.1, 0.5, 45.0, 0.2]
print("MCMC results")
for i in range(len(params)):
    print("{0} : {1} + {2} - {3}".format(labels[i], *params[i]))"""
    
paramsall = np.percentile(samples, [16, 50, 84], axis=0).T
params = np.zeros(paramsall.shape)
params[:,0] = paramsall[:,1]
params[:,1] = paramsall[:,2]-paramsall[:,1]
params[:,2] = paramsall[:,1]-paramsall[:,0]
#for i in range(params.shape[0]):
#    print("{0} : {1} + {2} - {3}".format(labels[i], *params[i]))
    
# plotting separate lorentzians. Calculate individual heights and freqs
height = 2.0 * params[0,0] **2 / (np.pi * params[2,0])
# Now calculate mode visibilities (shortened versions taken from Handberg and Campante (2011))
comp_m0 = np.cos(np.radians(params[-2,0]))**2
comp_m1 = 0.5 * np.sin(np.radians(params[-2,0]))**2
# Create array of mode heights for each component
heights = height * np.array([comp_m0, comp_m1, comp_m1])
freq_centres = np.array([params[1,0],params[1,0]+params[3,0],params[1,0]-params[3,0]+2.0*params[4,0]])

plt.figure('MCMC vs initial guess parameters')
plt.plot(freq, power, 'k', label='data', alpha=0.7)
plt.plot(freq, model_inst(params[:,0]), 'r', label='MCMC model') #model based on MCMC parametersplt.plot(freq, power, 'k')
plt.plot(freq, model_inst(p0), 'g', label='initial guess model') #actual model used to generate fake data
'''plt.plot(freq, lorentzian_individual(freq,heights[0],params[2,0],freq_centres[0]),'m')
plt.plot(freq, lorentzian_individual(freq,heights[1],params[2,0],freq_centres[1]),'m')
plt.plot(freq, lorentzian_individual(freq,heights[2],params[2,0],freq_centres[2]),'m')'''

plt.xlabel(r'Frequency ($\mu$Hz)', fontsize=18)
plt.ylabel(r'Power spectral density (ppm$^2$ Hz$^{-1}$)', fontsize=18)
#plt.title('MCMC vs initial guess parameters',fontsize=18)
plt.legend()
#================================================================================

line = 0.1*np.ones(len(freq)) # for 10% probability
S_nu = power/model_inst(params[:,0])
Nbins = len(freq)
probN = 1 - np.power(1-np.exp(-S_nu),Nbins) # equation 6.6, Chaplin and Basu 2016
plt.figure('check model, prob')
plt.plot(freq,probN,'k-')
plt.plot(freq, line, 'b--')
plt.ylim(0.0,1.1)
plt.xlim(freq[0], freq[-1])
plt.xlabel(r'Frequency ($\mu$Hz)', fontsize=18)
plt.ylabel(r'Probability', fontsize=18)
'''
line = np.ones(len(freq))
plt.figure('check model')
plt.plot(freq,S_nu,'k-')
plt.plot(freq, line, 'b--')
plt.xlabel(r'Frequency ($\mu$Hz)', fontsize=18)
plt.ylabel(r'data/model', fontsize=18)
'''
#================================================================================
#Correlation
# Create correlation map/plot
'''def cov2corr( A ):
    """
    covariance matrix to correlation matrix.
    """
    D = np.diag(np.sqrt(np.diag(A)))
    D_inv = np.linalg.inv(D)
    A = np.dot(D_inv, np.dot(A, D_inv))
    return A
# find covariance matrix to get correlation coefficients between parameters
covariance_mcmc = np.cov(samples[::10, :].T)
# Create correlation map
R_mcmc = cov2corr(covariance_mcmc)

fig, ax = plt.subplots()
heatmap = ax.pcolor(R_mcmc, cmap='Blues')
plt.colorbar(heatmap, label='Correlation Coefficient')
plt.yticks(np.arange(0.5, ndim+0.5), labels)
plt.xticks(np.arange(0.5, ndim+0.5), labels)
ax.get_xaxis().set_tick_params(direction='out', width=1)
ax.get_yaxis().set_tick_params(direction='out', width=1)
for y in range(R_mcmc.shape[0]):
    for x in range(R_mcmc.shape[1]):
        plt.text(x + 0.5, y + 0.5, '%.4f' % R_mcmc[y, x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )
                 
'''        

#=================================================================================
import csv 
#write parameters to text file called 'flattened_parameters_KIC8006161.txt'
with open(r'asymmetric_parameters_KIC8006161.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(params[:,0])'''