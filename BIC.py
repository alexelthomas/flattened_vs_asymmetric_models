#===============================================================================
#likelihood function
class likelihood(object):

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
        # But for Baye's factor we want the likelihood not log-likelihood so we 
        # return the inverse.
        like = np.sum(np.log(mod) + (self.power / mod))
        return np.exp(-1.0*like)
    
#Bayesian Information Criterion
class BIC(object):
    
    def __init__(self, _flike,_alike):
        self.flikelihood = _flike
        self.alikelihood = _alike  
        
    def __call__(self,params):
        """
        Computes Baye's factor given data and the two models and their best fitting parameters
        
        Inputs:
                :type flike:?
                :param flike: likelihood of flattened model
                
                :type alike:?
                :param alike: likelihood of asymmetric model
                
        Outputs:
                :type bayes: array-like
                :param bayes: Baye's factor                 

        """
        
        #separate parameters of two models
        fparams = params[:6]
        aparams = params[6:]
        
        fBIC = np.log(len(freq))*len(fparams) - 2*np.log(self.flikelihood(fparams))
        aBIC = np.log(len(freq))*len(aparams) - 2*np.log(self.alikelihood(aparams))
        
        return fBIC, aBIC
    
#model of Toutain 2008
class Model_flattened(object):
    
    def __init__(self, _freq):
        self.freq = _freq
    
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
        
        return m + background #modelled flattened lorentzian + background
    
    
#model with asymmetry
class Model_asymmetric(object):
    
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
    
    def __call__(self, params):
        """
        Sum of Lorentzian profiles providing a model for l=1 modes for a rms calibrated power spectrum

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
        amplitude, frequency, width, splitting, asymmetry, inclination, background = params

        # First calculate mode height
        height = 2.0 * amplitude **2 / (np.pi * width)

        # Now calculate mode visibilities (shortened versions taken from Handberg and Campante (2011))
        comp_m0 = np.cos(np.radians(inclination))**2
        comp_m1 = 0.5 * np.sin(np.radians(inclination))**2

        # Create array of mode heights for each component
        heights = height * np.array([comp_m0, comp_m1, comp_m1])
        
        # create components to make up the frequency for each mode.
        # will have central frequency + splitting depending on comps + asymmetry depending on compa        
        comps = np.array([0, 1, -1])
        compa = np.array([0, 0, 1]) #asymmetry component on m=-1 mode

        # Set up model array
        m = np.zeros(len(freq))
        for i in range(len(comps)):
            m += self.lorentzian([heights[i], width, frequency-(comps[i]*splitting)+(2.0*compa[i]*asymmetry)])
        return m + background #modelled lorentzians + background
    
#============================================================================================
 #load data ============================================================================
import numpy as np

# Read in parameters from both models and only take every third line to ignore the error values
flattened = np.genfromtxt('flattened_parameters_KIC8006161.csv',delimiter=',',skip_header=1)[::3]
asymmetric = np.genfromtxt('asymmetric_parameters_KIC8006161.csv',delimiter=',',skip_header=1)[::3]

#data = np.genfromtxt('kplr008006161_kasoc-wpsd_slc_v1.pow')
data = np.genfromtxt('kplr008006161_kasoc-wpsd_slc_v1.pow.txt')
allfrequency = data[:,0]
allpower = data[:,1]

bic = np.empty((np.shape(flattened)[0],3))

rangelength = 15.0

print ('If difference in BIC >2 then flattened model is preferred over asymmetry model.')

for j in range(np.shape(flattened)[0]):
    #Crop data to single l=1 multiplet
    freqrangestart =  flattened[j,1]
    
    freq_index= np.where(np.logical_and(allfrequency>freqrangestart , allfrequency<=freqrangestart+rangelength))[0]
    frequency1 = allfrequency[freq_index[0]:freq_index[-1]]
    power = allpower[freq_index[0]:freq_index[-1]]
    bw = allfrequency[1]-allfrequency[0]
    freq = np.arange(frequency1[0],frequency1[-1],bw)
    
    if len(freq)!=len(power):
        freq = np.append(freq,freq[-1]+bw)
    
    # Create model instance
    fmodel_inst = Model_flattened(freq)
    amodel_inst = Model_asymmetric(freq)
    
    # set up likelihood functions
    Flike = likelihood(freq, power, fmodel_inst)
    Alike = likelihood(freq, power, amodel_inst)
    
    # set up baye's factor instance using likelihood functions of two separate models
    bic_inst = BIC(Flike,Alike)
    
    #combine flattened and asymmetric parameters for one mode into a single array before calculating Baye's factor.
    params = np.hstack((flattened[j,:],asymmetric[j,:]))
    bic[j,0:2] = bic_inst(params)
    bic[j,2] = np.abs(bic[j,0]-bic[j,1])
    print ('for mode:',j+1,'Difference in BIC = ', bic[j,2])
    
