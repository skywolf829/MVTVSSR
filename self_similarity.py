from model import *
from options import *
from utility_functions import *
import torch.nn.functional as F
import torch
import os
import imageio
import argparse
from typing import Union, Tuple
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from skimage.transform.pyramids import pyramid_reduce
from skimage.feature import match_template
import cv2 as cv


MVTVSSR_folder_path = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(MVTVSSR_folder_path, "InputData")
output_folder = os.path.join(MVTVSSR_folder_path, "Output")
save_folder = os.path.join(MVTVSSR_folder_path, "SavedModels")

gauss_filter = np.array([[1, 4, 7, 4, 1],[4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4],[1, 4, 7, 4, 1]]) / 273

def compare_similarity(original_patch, new_patch):
    d = ((original_patch - new_patch)**2 * gauss_filter)
    return d.sum()

def create_required_similarity(frame, i, j):
    num_tested = 0
    req_sims = []

    # Check to the top left
    if(i > 0 and j > 0):
        test_frame = (frame[i:i+5, j:j+5] + frame[i-1:i+4, j-1:j+4]) / 2
        req_sims.append(compare_similarity(frame[i:i+5, j:j+5], test_frame))

    # Check to the bottom right
    if(i < frame.shape[0]-5 and j < frame.shape[1] - 5):
        test_frame = (frame[i:i+5, j:j+5] + frame[i+1:i+6, j+1:j+6]) / 2
        req_sims.append(compare_similarity(frame[i:i+5, j:j+5], test_frame))

    # Check to the top right
    if(i < frame.shape[0]-5 and j > 0):
        test_frame = (frame[i:i+5, j:j+5] + frame[i+1:i+6, j-1:j+4]) / 2
        req_sims.append(compare_similarity(frame[i:i+5, j:j+5], test_frame))

    # Check to the bottom left
    if(i > 0 and j < frame.shape[1] - 5):
        test_frame = (frame[i:i+5, j:j+5] + frame[i-1:i+4, j+1:j+6]) / 2
        req_sims.append(compare_similarity(frame[i:i+5, j:j+5], test_frame))

    # Check to the left
    if(i > 0):
        test_frame = (frame[i:i+5, j:j+5] + frame[i-1:i+4, j:j+5]) / 2
        req_sims.append(compare_similarity(frame[i:i+5, j:j+5], test_frame))

    # Check to the right
    if(i < frame.shape[0]-5):
        test_frame = (frame[i:i+5, j:j+5] + frame[i+1:i+6, j:j+5]) / 2
        req_sims.append(compare_similarity(frame[i:i+5, j:j+5], test_frame))

    # Check to top
    if(j > 0):
        test_frame = (frame[i:i+5, j:j+5] + frame[i:i+5, j-1:j+4]) / 2
        req_sims.append(compare_similarity(frame[i:i+5, j:j+5], test_frame))

    # Check to the bottom
    if(j < frame.shape[1] - 5):
        test_frame = (frame[i:i+5, j:j+5] + frame[i:i+5, j+1:j+6]) / 2
        req_sims.append(compare_similarity(frame[i:i+5, j:j+5], test_frame))

    return np.max(np.array(req_sims))

def sim_threshold(frame, i, j):
    req_sims = []

    f = frame[i:i+5, j:j+5]
    # Check to the top left
    if(i > 0 and j > 0):
        test_frame = (frame[i:i+5, j:j+5] + frame[i-1:i+4, j-1:j+4]) / 2
        req_sims.append(cv.matchTemplate(f, test_frame, cv.TM_SQDIFF))

    # Check to the bottom right
    if(i < frame.shape[0]-5 and j < frame.shape[1] - 5):
        test_frame = (frame[i:i+5, j:j+5] + frame[i+1:i+6, j+1:j+6]) / 2
        req_sims.append(cv.matchTemplate(f, test_frame, cv.TM_SQDIFF))

    # Check to the top right
    if(i < frame.shape[0]-5 and j > 0):
        test_frame = (frame[i:i+5, j:j+5] + frame[i+1:i+6, j-1:j+4]) / 2
        req_sims.append(cv.matchTemplate(f, test_frame, cv.TM_SQDIFF))

    # Check to the bottom left
    if(i > 0 and j < frame.shape[1] - 5):
        test_frame = (frame[i:i+5, j:j+5] + frame[i-1:i+4, j+1:j+6]) / 2
        req_sims.append(cv.matchTemplate(f, test_frame, cv.TM_SQDIFF))

    # Check to the left
    if(i > 0):
        test_frame = (frame[i:i+5, j:j+5] + frame[i-1:i+4, j:j+5]) / 2
        req_sims.append(cv.matchTemplate(f, test_frame, cv.TM_SQDIFF))

    # Check to the right
    if(i < frame.shape[0]-5):
        test_frame = (frame[i:i+5, j:j+5] + frame[i+1:i+6, j:j+5]) / 2
        req_sims.append(cv.matchTemplate(f, test_frame, cv.TM_SQDIFF))

    # Check to top
    if(j > 0):
        test_frame = (frame[i:i+5, j:j+5] + frame[i:i+5, j-1:j+4]) / 2
        req_sims.append(cv.matchTemplate(f, test_frame, cv.TM_SQDIFF))

    # Check to the bottom
    if(j < frame.shape[1] - 5):
        test_frame = (frame[i:i+5, j:j+5] + frame[i:i+5, j+1:j+6]) / 2
        req_sims.append(cv.matchTemplate(f, test_frame, cv.TM_SQDIFF))
    req_sims = np.array(req_sims)
    #print("min %.03f mean %.03f max %.03f" %(req_sims.min(), req_sims.mean(), req_sims.max()))
    return req_sims.max()


frame = imageio.imread(os.path.join(MVTVSSR_folder_path, "TestImage.jpg"))[:,:,0].astype(np.float32)
frame = np.load("ds.npy")[0].astype(np.float32)
print(frame.shape)

# 1. Gather all patches from the input and add it to a dictionary that counts 
# the number of similar patches, and calculate required similarity threshold

patches = []
patch_counts = []
patch_similarity_threshold = []

print("There will be %i patches to check" % ((frame.shape[0]-5)*(frame.shape[1]-5)))

count = 0
for i in range(frame.shape[0]-5):
    for j in range(frame.shape[1]-5):
        patches.append(frame[i:i+5, j:j+5])
        patch_similarity_threshold.append(sim_threshold(frame, i, j))



# 2. Iterate through the downscalings of the initial image
for downscale_iter in range(1, 7):
    factor = 1.25 ** downscale_iter
    f = pyramid_reduce(frame, downscale = factor).astype(np.float32)
    patch_counts_this_scale = []
    # 3. Iterate through each patch of the downscaled image
    print("There will be %i patches to check at this scale of %0.02f" % (((f.shape[0]-5)*(f.shape[1]-5)), factor))
    # 4. Compare similarity of the patch with all patches of the original image
    for k in range(len(patches)):
        sim = cv.matchTemplate(f, patches[k], cv.TM_SQDIFF).astype(np.float32)
        num_sim = np.where(sim <= patch_similarity_threshold[k], 1, 0).sum().astype(np.float32)
        patch_counts_this_scale.append(num_sim)
    patch_counts.append(patch_counts_this_scale)

all_hist_counts = []
# 6. Create histograms for counts
for k in range(len(patch_counts)):
    hist_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(patch_counts[k])):
        for j in range(len(hist_counts)):
            if(patch_counts[k][i] >= j):
                hist_counts[j] = hist_counts[j] + 1
    hist_counts = np.array(hist_counts).astype(np.float32)
    hist_counts *= (1 / hist_counts[0])
    all_hist_counts.append(hist_counts)
    print("Graph for scale " + str(1/(1.25**(k+1))) + ": " + str(hist_counts))


for k in range(len(all_hist_counts)):
    plt.plot(np.arange(0, all_hist_counts[k].shape[0], 1), all_hist_counts[k], label=str(1/(1.25**(k+1))))
plt.ylim(0, 1)
plt.xlabel("Num similar patches")
plt.ylabel("Percent")
plt.legend()

plt.show()
