import numpy as np
from astropy.modeling.models import Moffat2D
from astropy.modeling.models import custom_model
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.utils.exceptions import AstropyUserWarning
from astropy.coordinates import SkyCoord
import astropy.units as u
import re
import pathlib
import warnings
import subprocess


#Define the model
@custom_model
def MyModel2D(x, y, bg=5530, gamma=8.0, alpha=1.0, amp1=3e5, xc1=1674, yc1=1413, amp2=1e5, xc2=1192, yc2=406):

    #Create the model image. 
    model_im = np.zeros(x.shape)

    #Add the background
    model_im += bg

    #Add the Moffat profiles. 
    model_im += Moffat2D(amp1, xc1, yc1, gamma, alpha)(x,y)
    model_im += Moffat2D(amp2, xc2, yc2, gamma, alpha)(x,y)

    return model_im

class BSModel(object):

    def __init__(self, gmos_data_object):

        #Initialize values.
        self.gmos = gmos_data_object
        self.p = None

        #We are only interested in the two brightest stars. 
        bstars = np.loadtxt("list_of_bright_stars.txt")
        s = SkyCoord(bstars[:2,0]*u.deg, bstars[:2,1]*u.deg, frame='fk5')
        self.bspix = np.array(self.gmos.wcs.world_to_pixel(s)).T

        #Try to read the default model. 
        self.read()

        return

    #Fit the model
    def fit(self, **kwargs):

        #Set the grid to fit the model.
        y, x = np.mgrid[0:self.gmos.sci.data.shape[0], 0:self.gmos.sci.data.shape[1]]

        #Set the inital parameters.
        #self.p_init = MyModel2D(bg=bg, gamma=gamma, alpha=alpha, amp1=amp1, xc1=xc1, yc1=yc1, amp2=amp2, xc2=xc2, yc2=yc2)

        #Set the default initial parameters.
        self.p_init = MyModel2D(bg=self.gmos.median, gamma=8.0, alpha=1.0, amp1=3e5, xc1=self.bspix[0,0], yc1=self.bspix[0,1], amp2=1e5, xc2=self.bspix[1,0], yc2=self.bspix[1,1])

        #Now, set the parameters and initial conditions provided by the user. 
        for key in kwargs:
            par_name = re.sub("_",".",key)
            exec("self.p_init.{} = {}".format(par_name, kwargs[key]))

        #Run the fit.
        fit_p = LevMarLSQFitter()
        with warnings.catch_warnings():
            # Ignore model linearity warning from the fitter
            warnings.filterwarnings('ignore', message='Model is linear in parameters', category=AstropyUserWarning)
            self.p = fit_p(self.p_init, x[~self.gmos.mask], y[~self.gmos.mask], self.gmos.sci.data[~self.gmos.mask])

        return

    #Return the model image.
    def model_image(self):

        #Set the grid to for the model image.
        y, x = np.mgrid[0:self.gmos.sci.data.shape[0], 0:self.gmos.sci.data.shape[1]]        

        return self.p(x,y)

    #Save the model in a way that it is easy to read back.
    def save(self, model_fname=None, model_folder="models"):

        #Check the output folder exists.
        if not pathlib.Path(model_folder).exists():
            subprocess.call(["mkdir",model_folder])

        #Set the output filename. 
        if model_fname is None:
            model_fname = re.sub(".fits",".model",self.gmos.fname)

        #Write the model
        output_line = "MyModel2D("
        for key in self.p._param_metrics:
            output_line += "{}={},".format(key, eval("self.p."+key+".value"))
        output_line = output_line[:-1]+")"
        cat = open(model_folder+"/"+model_fname,"w")
        cat.write(output_line+"\n")
        cat.close()

        return 

    #Read a saved model.
    def read(self, model_fname=None, model_folder="models"):
        try:
            if model_fname is None:
                model_fname = re.sub(".fits",".model",self.gmos.fname)
            cat = open(model_folder+"/"+model_fname)
            self.p = eval(cat.readline()[:-1])
            cat.close()
        except FileNotFoundError:
            print("No model file found for image {}. Please run fit.".format(self.gmos.fname))
        return
