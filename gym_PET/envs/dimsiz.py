#get dimensions and size of image

def dimsiz(im):
    dim=len(im.shape)
    sz=im.shape
    if dim==2:
        fx, fy = im.shape
        print('two dimensions')
        sz= fx, fy
        return sz 
    else:
        fx, fy, fz = im.shape
        print('three dimensions')
        sz= fx, fy, fz
        return sz

sz=dimsiz(im)
bound=sz
vbnd=([20,50],[30,100]) #fxa fxb fya fyb

def boundbox2(im,bound,vbnd):
    if len(bound)==2:
        imc=im[vbnd[0]:vbnd[1],vbnd[2]:vbnd[3]]
        return imc
    else:
        imc=im[vbnd[0]:vbnd[1],vbnd[2]:vbnd[3]]
        return imc



