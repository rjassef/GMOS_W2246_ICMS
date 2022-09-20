import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.io import fits
from astropy.convolution import convolve, Gaussian2DKernel, interpolate_replace_nans
import re
import pathlib
import subprocess
import warnings

from gmosData import GMOSData
from bsModel import BSModel
from myparse import myparse

#Parse the command line arguments. 
args = myparse()

#Create the command line input folders if they do not exists. 
for folder in [args.icms_folder, args.corr_folder]:
    if not pathlib.Path(folder).exists():
        subprocess.call(["mkdir",folder])

#Read the data file names. 
cat = open(args.data_folder+"mrgfiles.txt")
mrg_files = [x[:-1] for x in cat.readlines()]
cat.close()

#There can be multiple dates, and we only want to generate a single median image per date. So first find the dates from the file names. 
dates = list()
for mrg_file in mrg_files:
    m = re.match("mrgS(\d{8})", mrg_file)
    if m.group(1) not in dates:
        dates.append(m.group(1))
print("Processing images for the following dates:",dates)

#Now, go through every file and fit the model to it if not done already. 
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
filters = np.array(filters)
print("Processing images for filters ", np.unique(filters))

#Now, get the median for each date if they do not yet exist.
print("Getting the median images")
median = dict()
for date in dates:
    for filter in np.unique(filters):

        if not args.recalculate_median and pathlib.Path(args.corr_folder+"median_{}_{}.msub.fits".format(date, filter)).exists():
            median[date+filter] = fits.getdata(args.corr_folder+"median_{}_{}.msub.fits".format(date, filter))
            continue

        use_date = np.zeros(all_data.shape[0], dtype=bool)
        for k, mrg_file in enumerate(mrg_files):
            if re.search(date,mrg_file):
                use_date[k] = True
        cond = (use_date) & (filters==filter)
        if all_data[cond].shape[0]==0:
            continue

        mean, median[date+filter], std = sigma_clipped_stats(all_data[cond], axis=0)#, mask=all_masks[use_date])
        
        #Save the mean and the median image. 
        fits.writeto(args.corr_folder+"mean_{}_{}.msub.fits".format(date, filter)  , mean  , overwrite=True)
        fits.writeto(args.corr_folder+"median_{}_{}.msub.fits".format(date, filter), median[date+filter], overwrite=True)
        fits.writeto(args.corr_folder+"std_{}_{}.msub.fits".format(date, filter)   , std   , overwrite=True)

#We no longer need the large arrays of all images and files, so remove them from memory.
del all_data
del all_masks

#Now, smooth the median images if not done already.
#If we do not have a smoothed version of the median, smooth it now. 
print("Smoothing the median images.")
smooth_median = dict()
for date in dates:
    for filter in np.unique(filters):
        if date+filter not in median:
            continue
        sm_name = "smoothed_median_{}_{}.msub.fits".format(date, filter)
        try:
            if args.recalculate_smooth_median:
                raise(FileNotFoundError)
            smooth_median[date+filter] = fits.getdata(args.corr_folder+sm_name)
        except FileNotFoundError:
            kernel = Gaussian2DKernel(x_stddev=10)
            smooth_median[date+filter] = convolve(median[date+filter], kernel)
            fits.writeto(args.corr_folder+sm_name, smooth_median[date+filter], overwrite=True)

#Finally apply the smoothed median as an illumination correction to all other images. 
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
