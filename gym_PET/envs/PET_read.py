#PET_read.py
#Read a PET image #2D for now and output a numpy array
#pip install nibabel
#http://nipy.org/nibabel/coordinate_systems.html#introducing-someone
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim

def get_mse(imref, it):
    imrs=imref.shape
    imts=it.shape 
    mx=np.maximum(imrs,imts)
    seg=np.zeros(mx)
    gt=np.zeros(mx)
    #cast both into the centre of largest
    dif=np.subtract(gt.shape,imts)
    diff=dif/2
    dh=np.floor(diff)
    if len(imrs)==3:
        seg[int(dh[0]):int(dh[0]+imts[0]), int(dh[1]):int(dh[1]+imts[1]), int(dh[2]):int(dh[2]+imts[2])]=it
        difr=np.subtract(gt.shape,imrs)
        diffr=difr/2
        dhr=np.floor(diffr)
        gt[int(dhr[0]):int(dhr[0]+imrs[0]), int(dhr[1]):int(dhr[1]+imrs[1]), int(dhr[2]):int(dhr[2]+imrs[2])]=imref
        err = np.sum((gt.astype("float") - seg.astype("float")) ** 2)
        err /= float(gt.shape[0] * gt.shape[1] * gt.shape[2])
        print("MSE is", err)
        return err
        
    else:
        seg[int(dh[0]):int(dh[0]+imts[0]), int(dh[1]):int(dh[1]+imts[1])]=it
        difr=np.subtract(gt.shape,imrs)
        diffr=difr/2
        dhr=np.floor(diffr)
        gt[int(dhr[0]):int(dhr[0]+imrs[0]), int(dhr[1]):int(dhr[1]+imrs[1])]=imref
        err = np.sum((gt.astype("float") - seg.astype("float")) ** 2)
        err /= float(gt.shape[0] * gt.shape[1])
        print("MSE is", err)
        return err

def get_structsim(imref, it):
    imrs=imref.shape
    imts=it.shape 
    mx=np.maximum(imrs,imts)
    seg=np.zeros(mx)
    gt=np.zeros(mx)
    #cast both into the centre of largest
    dif=np.subtract(gt.shape,imts)
    diff=dif/2
    dh=np.floor(diff)
    if len(imrs)==3:
        seg[int(dh[0]):int(dh[0]+imts[0]), int(dh[1]):int(dh[1]+imts[1]), int(dh[2]):int(dh[2]+imts[2])]=it
        difr=np.subtract(gt.shape,imrs)
        diffr=difr/2
        dhr=np.floor(diffr)
        gt[int(dhr[0]):int(dhr[0]+imrs[0]), int(dhr[1]):int(dhr[1]+imrs[1]), int(dhr[2]):int(dhr[2]+imrs[2])]=imref
        strsim=ssim(gt,seg)
        print("structed similarity is", strsim)
        return strsim 
        
    else:
        seg[int(dh[0]):int(dh[0]+imts[0]), int(dh[1]):int(dh[1]+imts[1])]=it
        difr=np.subtract(gt.shape,imrs)
        diffr=difr/2
        dhr=np.floor(diffr)
        gt[int(dhr[0]):int(dhr[0]+imrs[0]), int(dhr[1]):int(dhr[1]+imrs[1])]=imref
        strsim=ssim(gt,seg)
        print("structed similarity is", strsim)
        return strsim 



def get_ref_for_dice():
    bl=[30, 30, 30]
    #For now set the reference as a bounding box 25x25x25 in the middle of image
    PET_img = nib.load('/home/petic/anaconda/envs/A3Cexample/TestPet/gym_PET/envs/io/test.nii', mmap=False)
    PET_img_data = PET_img.get_data()
    sh=PET_img_data.shape
    imm=PET_img_data
    #imm=np.random.random((3,3,3))
    im=imm.squeeze()
    sz=im.shape
    dm=len(sz)
    if dm == 2:
        fa=round(im.shape[0]/2)-bl[0]/2
        faa=round(im.shape[0]/2)+bl[0]/2
        fb=round(im.shape[1]/2)-bl[1]/2
        fbb=round(im.shape[1]/2)+bl[1]/2
        imref=im[fa:faa,fb:fbb]
        return imref
    else:
        fa=int(round(im.shape[0]/2)-bl[0]/2)
        faa=int(round(im.shape[0]/2)+bl[0]/2)
        fb=int(round(im.shape[1]/2)-bl[1]/2)
        fbb=int(round(im.shape[1]/2)+bl[1]/2)
        fc=int(round(im.shape[1]/2)-bl[2]/2)
        fcc=int(round(im.shape[1]/2)+bl[2]/2)
        imref=im[fa:faa, fb:fbb, fc:fcc]
        return imref



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

