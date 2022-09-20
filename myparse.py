import argparse

def myparse():

    parser = argparse.ArgumentParser(description='Remove the glare from the bright stars and apply an illumination correction.')

    #Folders
    parser.add_argument('--data_folder', default='orig/', metavar='folder_name', type=str, help='Folder with the pipeline reduced mrg files.')
    parser.add_argument('--icms_folder', default='icms/', metavar='folder_name', type=str, help='Folder with the pipeline reduced mrg files.')
    parser.add_argument('--corr_folder', default='illumination_images/', metavar='folder_name', type=str, help='Folder with the pipeline reduced mrg files.')
    parser.add_argument('--Ncpu', type=int, help='Number of CPUs to use')

    #Flow control options. 
    parser.add_argument('--recalculate_masks', action='store_true')
    parser.add_argument('--recalculate_fits' , action='store_true')
    parser.add_argument('--recalculate_median', action='store_true')
    parser.add_argument('--skip_mask_convolution', action='store_true')
    parser.add_argument('--recalculate_smooth_median', action='store_true')

    #Parse the command line arguments
    args = parser.parse_args()

    return args
