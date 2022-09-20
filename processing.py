import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve, Gaussian2DKernel, interpolate_replace_nans
import warnings 
import re

from gmosData import GMOSData
from bsModel import BSModel

def initial_processing(args, mrg_files):

    for k, mrg_file in enumerate(mrg_files):

        #Read the image. 
        gmos = GMOSData(mrg_file, recalculate_mask=args.recalculate_masks, data_folder=args.data_folder)

        #Save the filter. 
        if k==0:
            filters = list()
        filters.append(gmos.filter)

        #Define the arrays to hold all the data to determine the median image 
        if k==0:   
            all_data  = np.zeros((len(mrg_files), gmos.sci.data.shape[0], gmos.sci.data.shape[1]))
            all_masks = np.zeros(all_data.shape, dtype=bool)

        #Fit the model.
        model = BSModel(gmos)
        if model.p is None or args.recalculate_fits:
            model.fit(amp1=1e5, amp2=4e4, gamma_fixed=True, alpha_fixed=True)
            model.save()

        #Save the image after removing the glow of the bright stars, but not the background.
        im_diff = gmos.sci.data - model.model_image() + model.p.bg

        #Save the normalized data with the bright stars removed and the mask and close the image.
        _ , norm, _ = sigma_clipped_stats(im_diff, mask=gmos.mask)
        if args.skip_mask_convolution:
            all_data[k]  = im_diff/norm
            all_data[k,gmos.mask] = 1.0
        else:
            im_diff /= norm
            im_diff[gmos.mask] = np.nan
            kernel = Gaussian2DKernel(x_stddev=1, x_size=17, y_size=17)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                im_diff_interp = interpolate_replace_nans(im_diff, kernel)
            all_data[k] = np.where(gmos.mask, im_diff_interp, im_diff)
            all_masks[k] = gmos.mask
            del(model)
            del(gmos)

    return all_data, all_masks, filters


def final_processing(args, median, smooth_median, mrg_files):

    for mrg_file in mrg_files:

        #Get the date
        m = re.match("mrgS(\d{8})", mrg_file)
        date = m.group(1)

        #Load the image.
        gmos = GMOSData(mrg_file, data_folder=args.data_folder)

        #Remove the glow of the bright stars. 
        model = BSModel(gmos)
        gmos.sci.data -= (model.model_image() - model.p.bg)

        #Apply the correction. 
        if gmos.filter[0]=='z':
            new_sci = gmos.sci.data/median[date+gmos.filter]
        else:
            new_sci = gmos.sci.data/smooth_median[date+gmos.filter]

        #Now, update the DQ field to mask out the places where the smooth median is nan or exceedingly small. 
        new_dq = np.where((gmos.dq.data==0) & ((np.isnan(median[date+gmos.filter]))| (median[date+gmos.filter]<1e-5)), 513, gmos.dq.data)

        #Save the resulting file.
        fname_out = re.sub(".fits","_icms.fits",mrg_file)
        gmos.save_updated_image(fname_out, sci_data=new_sci,dq_data=new_dq, overwrite=True, save_folder=args.icms_folder)

        #Delete the object. 
        del(gmos)

    return

def mp_sigma_clipped_stats(axis, data):
    return sigma_clipped_stats(data, axis=axis)
