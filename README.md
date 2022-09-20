# GMOS_W2246_ICMS

Results to be published in Zewdie et al. (2022).

This code subtracts the glow of the bright stars and to correct for fringing and uneven bagckgrounds in the deep GMOS imaging of W2246-0526.

To run this code, put all of the mrg images (reduced with the IRAF gemini package) in a folder call orig/. Make a list of all the mrg images in a file called mrgfiles.txt. 

Then, run the [additional_corrections_mp.py](additional_corrections_mp.py) script. The script [additional_corrections.py](additional_corrections.py) should return the same results, but does not use parallel processing. 

To see all the command line arguments, call the script with the --help flag. 