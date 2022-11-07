import logging  # when to use what type of logging:

# set root logger one level further down to INFO to enable info display in Terminal: https://stackoverflow.com/questions/11548674/logging-info-doesnt-show-up-on-console-but-warn-and-error-do
logging.getLogger().setLevel(
    logging.INFO)  # https://stackoverflow.com/questions/2031163/when-to-use-the-different-log-levels
import os
import pandas as pd
import cv2
import glob

# ---------------------------------------------------------------------------- #
#                                   Functions                                  #
# ---------------------------------------------------------------------------- #

def load_sample_data() -> pd.DataFrame():
    # from local resources
    images = [cv2.imread(file) for file in glob.glob("./small_sample_out/*.jpeg")]
    metadata = pd.read_hdf('./small_sample_out/metadata_sample.hdf', '/d')

    # check if some data is loaded
    #assert len(images()) == 0, "Images available"

    return images, metadata

# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #


