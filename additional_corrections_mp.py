import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.io import fits
from astropy.convolution import convolve, Gaussian2DKernel, interpolate_replace_nans
import re
import pathlib
import subprocess
import warnings
import multiprocessing as mp
from functools import partial

from gmosData import GMOSData
from bsModel import BSModel
from myparse import myparse

from processing import initial_processing, final_processing, mp_sigma_clipped_stats

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

#Define the number of CPUs to use. 
if args.Ncpu is None:
    args.Ncpu = np.min([mp.cpu_count(), len(mrg_files)])

#Setup the functions to run in mp. 
Pool = mp.Pool(args.Ncpu)
func = partial(initial_processing, args)

#Run the initial processing in mp.
mrg_files_split = np.array_split(mrg_files, args.Ncpu)
Output = Pool.map(func, mrg_files_split)

#Format the output.
all_data = list()
all_masks = list()
filters = list()
for op in Output:
    all_data.extend(op[0])
    all_masks.extend(op[1])
    filters.extend(op[2])
all_data = np.array(all_data)
all_masks = np.array(all_masks)
filters = np.array(filters)
print("Processing images for filters ", np.unique(filters))

args.recalculate_median = True

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

        #mean, median[date+filter], std = sigma_clipped_stats(all_data[cond], axis=0)#, mask=all_masks[use_date])
        Pool = mp.Pool(args.Ncpu)
        func = partial(mp_sigma_clipped_stats, 0)
        all_data_split = np.array_split(all_data[cond], args.Ncpu, axis=2)
        Output = Pool.map(func, all_data_split)
  
        aux_mean = list()
        aux_median = list()
        aux_std = list()
        for op in Output:
            aux_mean.append(op[0])
            aux_median.append(op[1])
            aux_std.append(op[2])
        mean = np.hstack(aux_mean)
        median[date+filter] = np.hstack(aux_median)
        std = np.hstack(aux_std)

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
Pool = mp.Pool(args.Ncpu)
func = partial(final_processing, args, median, smooth_median)
mrg_files_split = np.array_split(mrg_files, args.Ncpu)
Pool.map(func, mrg_files_split)
