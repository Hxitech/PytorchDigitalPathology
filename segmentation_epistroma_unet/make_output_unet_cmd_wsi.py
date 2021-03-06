
# coding: utf-8

# v2
# 7/11/2018

# In[ ]:


import argparse
import os
import glob
import numpy as np
import cv2
import torch
import sys
import time
import math
from pathlib import Path

from WSI_handling import wsi
from unet import UNet

from shapely.geometry import Polygon


# In[ ]:


#-----helper function to split data into batches
def divide_batch(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 


# In[ ]:


# ----- parse command line arguments
parser = argparse.ArgumentParser(description='Make output for entire image using Unet')
parser.add_argument('input_pattern',
                    help="input filename pattern. try: *.png, or tsv file containing list of files to analyze",
                    nargs="*")


# In[ ]:


parser.add_argument('-r', '--resolution', help="image resolution in microns per pixel", default=1, type=float)
parser.add_argument('-c', '--color', help="annotation color to use, default None", default='green', type=str)
parser.add_argument('-a', '--annotation', help="annotation index to use, default largest", default='wsi', type=str)

parser.add_argument('-p', '--patchsize', help="patchsize, default 256", default=256, type=int)
parser.add_argument('-s', '--batchsize', help="batchsize for controlling GPU memory usage, default 10", default=10, type=int)
parser.add_argument('-o', '--outdir', help="outputdir, default ./output/", default="./output/", type=str)
parser.add_argument('-m', '--model', help="model", default="best_model.pth", type=str)
parser.add_argument('-i', '--gpuid', help="id of gpu to use", default=0, type=int)
parser.add_argument('-f', '--force', help="force regeneration of output even if it exists", default=False,
                    action="store_true")
parser.add_argument('-b', '--basepath',
                    help="base path to add to file names, helps when producing data using tsv file as input",
                    default="", type=str)


# In[ ]:


args = parser.parse_args()


# In[ ]:


if not (args.input_pattern):
    parser.error('No images selected with input pattern')


# In[ ]:


OUTPUT_DIR = args.outdir


# In[ ]:


batch_size = args.batchsize
patch_size = args.patchsize
base_stride_size = patch_size//2


# In[ ]:


# ----- load network
device = torch.device(args.gpuid if torch.cuda.is_available() else 'cpu')


# In[ ]:


checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage) #load checkpoint to CPU and then put to device https://discuss.pytorch.org/t/saving-and-loading-torch-models-on-2-machines-with-different-number-of-gpu-devices/6666
model = UNet(n_classes=checkpoint["n_classes"], in_channels=checkpoint["in_channels"],
             padding=checkpoint["padding"], depth=checkpoint["depth"], wf=checkpoint["wf"],
             up_mode=checkpoint["up_mode"], batch_norm=checkpoint["batch_norm"]).to(device)
model.load_state_dict(checkpoint["model_dict"])
model.eval()


# In[ ]:


print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")


# ----- get file list

# In[ ]:


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# In[ ]:


files = []
basepath = args.basepath  #
basepath = basepath + os.sep if len(
    basepath) > 0 else ""  # if the user supplied a different basepath, make sure it ends with an os.sep


# In[ ]:


if len(args.input_pattern) > 1:  # bash has sent us a list of files
    files = args.input_pattern
elif args.input_pattern[0].endswith("tsv"):  # user sent us an input file
    # load first column here and store into files
    with open(args.input_pattern[0], 'r') as f:
        for line in f:
            if line[0] == "#":
                continue
            files = line.split("\t")
else:  # user sent us a wildcard, need to use glob to find files
    files = glob.glob(args.basepath + args.input_pattern[0])


# In[ ]:


def run_model(img_dims,patch_size,stride_size,base_stride_size,batch_size,args,img,annotation):
    x_start = int(img_dims[0])
    y_start = int(img_dims[1])
    w_orig = img.get_coord_at_mpp(img_dims[2] - x_start,input_mpp=img['mpp'],output_mpp=args.resolution)
    h_orig = img.get_coord_at_mpp(img_dims[3] - y_start,input_mpp=img['mpp'],output_mpp=args.resolution)

    w = int(w_orig + (patch_size - (w_orig % patch_size)))
    h = int(h_orig + (patch_size - (h_orig % patch_size)))

    base_edge_length = base_stride_size*int(math.sqrt(batch_size))        
    
    # need to make sure we don't end up with a last row/column smaller than patch_size
    h = h + patch_size if base_edge_length - (h % base_edge_length) else h
    w = w + patch_size if base_edge_length - (w % base_edge_length) else w

    roi = img.get_tile(args.resolution,(x_start-stride_size//2,y_start-stride_size//2),(w+base_stride_size,h+base_stride_size))
    x_points = range(0,np.shape(roi)[0],base_stride_size*int(math.sqrt(batch_size)))
    y_points = range(0,np.shape(roi)[1],base_stride_size*int(math.sqrt(batch_size)))
    grid_points = [(x,y) for x in x_points for y in y_points]                

    output = np.zeros([np.shape(roi)[0],np.shape(roi)[1]],dtype='uint8')

    for i,batch_points in enumerate(grid_points):

        # get the tile of the batch
        big_patch = roi[batch_points[0]:(batch_points[0]+base_edge_length+base_stride_size),batch_points[1]:(batch_points[1]+base_edge_length+base_stride_size),:]

        big_patch_gpu = torch.from_numpy(big_patch).type('torch.FloatTensor').to(device)
        # split the tile into patch_size patches
        batch_arr = torch.stack(([big_patch_gpu[x:x+patch_size,y:y+patch_size,:] for y in range(0,np.shape(big_patch_gpu)[1]-base_stride_size,base_stride_size) for x in range(0,np.shape(big_patch_gpu)[0]-base_stride_size,base_stride_size)]))        
        batch_arr = batch_arr.permute(0,3,1,2) / 255

        # ---- get results
        output_batch = model(batch_arr)
        output_batch = output_batch.argmax(axis=1)

        #remove the padding from each tile, we only keep the center            
        output_batch = output_batch[:,base_stride_size//2:-base_stride_size//2,base_stride_size//2:-base_stride_size//2]            

        # --- pull from GPU and append to rest of output 
        output_batch = output_batch.detach().cpu().numpy()            

        reconst = np.concatenate(np.concatenate(output_batch.reshape(int(np.shape(big_patch)[1]/(patch_size//2))-1,int(np.shape(big_patch)[0]/(patch_size//2))-1,base_stride_size,base_stride_size),axis=2),axis=0)

        output[batch_points[0]:(batch_points[0]+np.shape(big_patch)[0]-base_stride_size),batch_points[1]:(batch_points[1]+np.shape(big_patch)[1]-base_stride_size)] = reconst


    if(args.annotation.lower() != 'wsi'):
    #in case there was extra padding to get a multiple of patch size, remove that as well
        _,mask = img.get_annotated_region(args.resolution,args.color,annotation,return_img=False)            
        output = output[0:mask.shape[0], 0:mask.shape[1]] #remove paddind, crop back
        output = np.bitwise_and(output>0,mask>0)*255
    
    return output

for fname in files:    
    fname = fname.strip()
    
    if(args.annotation.lower() != 'all'):        
    
        newfname_class = "%s/%s_class.png" % (OUTPUT_DIR, Path(fname).stem)

        if not args.force and os.path.exists(newfname_class):
            print("Skipping as output file exists")
            continue
        print(f"working on file: \t {fname}")
        print(f"saving to : \t {newfname_class}")

        start_time = time.time()
        cv2.imwrite(newfname_class, np.zeros(shape=(1, 1)))                                            

    xml_fname = Path(fname).with_suffix('.xml')
    if not os.path.exists(xml_fname):
        xml_fname = Path(fname).with_suffix('.json')

    if os.path.exists(xml_fname):
        img = wsi(fname,xml_fname)
        stride_size = int(base_stride_size * (args.resolution/img["mpp"]))

        if(args.annotation.lower() == 'all'):        
            annotations_todo = len(img.get_points(args.color,[]))
            print(f"working on file: \t {fname}")            

            for k in range(0,annotations_todo):                
                print('Working on annotation ' + str(k))
                start_time = time.time()
                img_dims = img.get_dimensions_of_annotation(args.color,k)

                newfname_class = "%s/%s_%d_class.png" % (OUTPUT_DIR, Path(fname).stem,k)

                if args.force or not os.path.exists(newfname_class):
                    output = run_model(img_dims,patch_size,stride_size,base_stride_size,batch_size,args,img,annotation=k)        
                    cv2.imwrite(newfname_class, output)                

                output = None
                print('Elapsed time = ' + str(time.time()-start_time))

        else:            

            if(args.annotation.lower() == 'wsi'):
                img_dims = [0,0,img["img_dims"][0][0],img["img_dims"][0][1]]
            else:
                img_dims = img.get_dimensions_of_annotation(args.color,args.annotation)

            if img_dims:
                output = run_model(img_dims,patch_size,stride_size,base_stride_size,batch_size,args,img,annotation=args.annotation)        
                cv2.imwrite(newfname_class, output)
                output = None
                print('Elapsed time = ' + str(time.time()-start_time))

            else:
                print('No annotation of color')
    else:
        print('Could not find ' + str(xml_fname))      