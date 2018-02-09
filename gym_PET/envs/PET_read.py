#PET_read.py
#Read a PET image #2D for now and output a numpy array
#pip install nibabel
#http://nipy.org/nibabel/coordinate_systems.html#introducing-someone
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def get_data():
    PET_img = nib.load('/home/petic/anaconda/envs/A3Cexample/TestPet/gym_PET/envs/io/test.nii', mmap=False)
    PET_img_data = PET_img.get_data()
    PET_img_data.shape
    imm=PET_img_data
    #imm=np.random.random((3,3,3))
    im=imm.squeeze()
    return im

def get_siz():
    PET_img = nib.load('/home/petic/anaconda/envs/A3Cexample/TestPet/gym_PET/envs/io/test.nii', mmap=False)
    PET_img_data = PET_img.get_data()
    PET_img_data.shape
    imm=PET_img_data
    #imm=np.random.random((3,3,3))
    im=imm.squeeze()
    sz=im.shape
    dm=len(sz)
    if dm == 2:
        fx, fy = im.shape
        print('two dimensions')
        sz= fx, fy
        return sz
    else:
        fx, fy, fz = im.shape
        print('three dimensions')
        sz= fx, fy, fz
        return sz


def show_slices(slices, fig, axes):
    """ Function to display row of image slices """
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
                #plt.show()
    plt.draw()
    return axes


#slice_0 = PET_img_data[125, :, :,0]
#slice_1 = PET_img_data[:, 160, :,0]
#slice_2 = PET_img_data[:, :, 100,0]

#show_slices([slice_0, slice_1, slice_2])
#plt.suptitle("Center slices for EPI image")

