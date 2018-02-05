#PET_read.py
#Read a PET image #2D for now and output a numpy array
#pip install nibabel
#http://nipy.org/nibabel/coordinate_systems.html#introducing-someone
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def get_data():
	PET_img = nib.load('/home/putz/anaconda3/envs/TestPet/TestPet/gym_PET/envs/io/test.nii')
	PET_img_data = PET_img.get_data()
	PET_img_data.shape
	return PET_img_data

def show_slices(slices):
	""" Function to display row of image slices """
	fig, axes = plt.subplots(1, len(slices))
	for i, slice in enumerate(slices):
		axes[i].imshow(slice.T, cmap="gray", origin="lower")
                #plt.show()

#slice_0 = PET_img_data[125, :, :,0]
#slice_1 = PET_img_data[:, 160, :,0]
#slice_2 = PET_img_data[:, :, 100,0]

#show_slices([slice_0, slice_1, slice_2])
#plt.suptitle("Center slices for EPI image")

