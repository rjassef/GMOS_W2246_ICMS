import numpy as np 
import warnings
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from regions.shapes.circle import CirclePixelRegion
from regions.core import PixCoord
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, LinearStretch, ImageNormalize

class GMOSData(object):

    def __init__(self, fname, recalculate_mask=False, data_folder="."):
        
        #Start by loading the image.
        self.fname = fname
        self.data_folder = data_folder
        h = fits.open(self.data_folder+"/"+self.fname)
        self.sci = h['SCI']
        self.var = h['VAR']
        self.dq  = h['DQ']
        self.filter = h[0].header['FILTER2']

        #Get basic stats.
        self.mean, self.median, self.std = sigma_clipped_stats(self.sci.data, sigma=3.0)

        #Save the WCS
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            self.wcs = WCS(self.sci.header)

        #See if the mask is in the HDU already. If not, create the mask.
        if 'MASK' in h and not recalculate_mask:
            self.mask = h['MASK'].data.astype(bool)
        else:

            print("No mask in HDU of {}. Creating it now.".format(fname))

            #Set up the initial mask by combining the DQ mask with a mask blocking every pixel above 50000. 
            self.mask = np.where((self.dq.data>0) | (self.sci.data>50000), True, False)

            #Now detect and mask all the sources. 
            self.mask_sources()

            #Mask the guider probe in the r-band. It is always in the same position for all images. Not present in the i-band.
            #if self.filter == 'r_G0326':
            #    self.mask[0:210,2325:2582] = True

            #Now, mask the very bright stars. 
            self.mask_bright_stars()

            #Finally, mask some bad columns that can become problematic. 
            #self.mask[:, 740: 742] = True
            #self.mask[:,1170:1179] = True
            #self.mask[:,1395:1398] = True
            #self.mask[:,1229:1232] = True
            #self.mask[:,1710:1714] = True
            self.mask[:, 738: 744] = True
            self.mask[:,1168:1181] = True
            self.mask[:,1393:1400] = True
            self.mask[:,1227:1234] = True
            self.mask[:,1708:1716] = True

            #Save it with the new mask.
            self.save_updated_image(self.fname, overwrite=True)

        return

    #Convenience function to mask the stars.
    def add_star_mask(self, xc, yc, rad):
        if np.isscalar(rad):
            rad = rad*np.ones(len(xc))
        for k in range(len(xc)):
            center = PixCoord(xc[k],yc[k])
            reg = CirclePixelRegion(center, rad[k])
            if reg.to_mask().to_image(self.sci.data.shape) is not None:
                self.mask += reg.to_mask().to_image(self.sci.data.shape).astype(bool)
        return

    def mask_sources(self): 
        daofind = DAOStarFinder(fwhm=8.0, threshold=5.*self.std)
        sources = daofind(self.sci.data - self.median)
        self.add_star_mask(sources['xcentroid'], sources['ycentroid'], 15.0)
        return

    def mask_bright_stars(self):
        bstars = np.loadtxt("list_of_bright_stars.txt")
        s = SkyCoord(bstars[:,0]*u.deg, bstars[:,1]*u.deg, frame='fk5')
        pix = np.array(self.wcs.world_to_pixel(s)).T
        self.add_star_mask(pix[:,0], pix[:,1], bstars[:,2])
        return 

    def save_updated_image(self, outname, sci_data=None, var_data=None, dq_data=None, overwrite=False, save_folder=None):
        #Open the original image.
        h = fits.open(self.data_folder+"/"+self.fname)

        #Update the SCI and VAR extensions if needed. 
        if sci_data is not None:
            h['SCI'].data = sci_data
        if var_data is not None:
            h['VAR'].data = var_data
        if dq_data is not None:
            h['DQ'].data = dq_data

        #See if the mask exists already. If not, add that dimension.
        if 'MASK' not in h:
            h.append(fits.hdu.image.ImageHDU())
            h[-1].header['EXTNAME'] = 'MASK'
        
        #Add the updated mask. 
        h['MASK'].data = self.mask.astype(np.int32)

        #Save the fits file. 
        if save_folder is None:
            save_folder = self.data_folder
        h.writeto(save_folder+"/"+outname, overwrite=overwrite)
        h.close()

        return

    def display(self, with_mask=False):
        norm = ImageNormalize(self.sci.data, interval=ZScaleInterval(), stretch=LinearStretch())

        if with_mask:
            plt.imshow(np.where(self.mask, 0, self.sci.data), norm=norm, cmap='gray', origin='lower')
        else:
            plt.imshow(self.sci.data, norm=norm, cmap='gray', origin='lower')
        
        plt.show()

        return
        
    
