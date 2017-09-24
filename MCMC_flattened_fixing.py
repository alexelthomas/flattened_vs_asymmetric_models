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
        if not np.isfinite(lp): #check if equal to infinity or nan
            return -np.inf
        return lp + self.like(params)
    
#model
class Model(object):
    
    def __init__(self, _freq):
        self.freq = _freq
        
    '''def lorentzian(self, params):
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

        return height / (1.0+ (4.0 / width**2)*(self.freq - frequency)**2)'''
    
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
        amplitude, frequency, width, splitting, inclination, background = params

        # First calculate mode height
        height = 2.0 * amplitude **2 / (np.pi * width)

        # Set up model array
        m = np.zeros(len(freq))
        epsilon = splitting/width
        X = (self.freq-(frequency+0.5*splitting))/(0.5*width)
        denominator = 2.0*epsilon/(1.-epsilon**2+X**2)
        
        # to get flat top rather than drop in middle of model we must make sure that arctan(*)>0
        m = height/(2.0*epsilon)*np.arctan(np.abs(denominator))
        #m[np.where(m<0.)[0]] = height/(2.0*epsilon)*0.5*np.pi
            
        
        #m += self.lorentzian([heights[i], width, frequency-(comps[i]*splitting)+(2.0*compa[i]*asymmetry)])
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
    #data = limit_spectrum
    return data, limit_spectrum
 
#def lorentzian_individual(freq, height, width, cent_freq):
 #       return height / (1.0+ (4.0 / width**2)*(freq - cent_freq)**2)
    
    
#Generate data ==========================================================================
# Create frequency array
# bin width for 4-years of Kepler in uHz
bw = 1.0 / (4.0 * 365.0 * 86400.0) * 1e6
freq = np.arange(2990,3010, bw)

# Create model instance
model_inst = Model(freq)

# Parameters for artificial data
# amplitude, frequency, width, splitting, inclination, background
p0 = [20.0, 3000.0, 1.0, 0.1, 90.0, 0.0]
p1 = [20.0, 3000.0, 1.0, 1.1, 90.0, 0.0]

# Generate fake data and plot points as well as model used for generation
power, limit_model = generate_data(freq, p0, model_inst)
model0 = model_inst(p0)
model1 = model_inst(p1)

plt.figure('Generated data and true model')
plt.plot(freq, power, 'k')
plt.plot(freq, model0, 'r')
plt.plot(freq, model1, 'b')
plt.yscale('log')
plt.xlabel(r'Frequency ($\mu$Hz)', fontsize=18)
plt.ylabel(r'PSD (arb. units)', fontsize=18)

'''
 #load data ============================================================================
data = np.genfromtxt('kplr008006161_kasoc-wpsd_slc_v1.pow')
allfrequency = data[:,0]
allpower = data[:,1]

#Crop data to single l=1 multiplet
rangelength = 10.0
freqrangestart =  3138.0

i=0
while allfrequency[i]<freqrangestart:
    i+=1
minfreq=i
i=0
while allfrequency[i]<freqrangestart+rangelength:
    i+=1
maxfreq=i
                
frequency1 = allfrequency[minfreq:maxfreq]
power = allpower[minfreq:maxfreq]
bw = allfrequency[1]-allfrequency[0]

freq = np.arange(frequency1[0],frequency1[-1],bw)

if len(freq)!=len(power):
    freq = np.append(freq,freq[-1]+bw)

#frequency1 = allfrequency[minfreq:maxfreq-1]

# Create model instance
model_inst = Model(freq)
'''
'''plt.figure('Data')
plt.plot(freq, power, 'k')
plt.xlabel(r'Frequency ($\mu$Hz)', fontsize=18)
plt.ylabel(r'PSD (arb. units)', fontsize=18)
''''''

#==================================================================================
# SET UP PRIORS
# Create prior, likelihood instances
# amplitude, frequency, width, splitting, inclination, background
#bounds = [(0.0, 20000.0), (179.0, 180.0), (0.0, 0.5), (0.0, 1.0), (-1.0,1.0), (0.0, 90.0), (0.0, 200.0)]
#for p modes
#bounds = [(0.0, 15000.0), (160.6,161.3), (0.0, 0.5), (0.0, 1.0), (-1.0,1.0), (0.0, 90.0), (0.0, 500.0)]
bounds = [(0.0, 100.0), (3141.0,3144.0), (0.0,5.0), (0.0, 3.0), (0.0, 90.0), (0.0, 100.0)]
Prior = lnprior(bounds)
Lnlike = lnlike_mcmc(freq, power, model_inst)
Lnprob = lnprob(Lnlike, Prior)

# =================================================================================
#FIND FIRST GUESSES

# Smooth data first, less susceptible to noise
import scipy.ndimage
# smooth with Gaussian kernel convolution
power_smooth = scipy.ndimage.filters.gaussian_filter1d(power,20.0)
# Central frequency estimate from frequency of highest peak in region
central_freq_est = freq[np.argmax(power_smooth)]

#splitting_est = np.mean(np.diff(peak_freqs))
splitting_est = 0.6

#amplitude_est = np.sqrt(m0_height/2.0) #ignores factor of width*pi
amplitude_est = 2.0

# Width
width_est = 1.5

# Inclination angle
angle_est = 45.0

# Background can be worked out as taking average of bins in extrema of
# fitting region
numbins = 30
background_est = np.mean(np.r_[power[:numbins], power[-numbins:]])

# amplitude, frequency, width, splitting, inclination, background
# First guess parameters
p0 = [amplitude_est, central_freq_est, width_est, splitting_est,angle_est, background_est]

plt.figure('First guess parameters')
plt.plot(freq, power, 'k')
plt.plot(freq, model_inst(p0), 'r') #model based on first guess parameters
#plt.plot(freq, model_inst(trueparams), 'g') #actual model used to generate fake data
#plt.plot(freq, power_smooth, 'r')
#plt.plot(freq[peaks][idx], power_smooth[peaks][idx], 'rD')
plt.xlabel(r'Frequency ($\mu$Hz)', fontsize=18)
plt.ylabel(r'PSD (arb. units)', fontsize=18)
#plt.xlim(4, 6)
#plt.ylim(0, 10)
'''