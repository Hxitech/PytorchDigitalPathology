
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
import sklearn.feature_extraction.image
import sys

sys.path.insert(1,'/mnt/data/home/pjl54/WSI_handling')
import wsi


# In[ ]:


from unet import UNet


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
            files.append(basepath + line.strip().split("\t")[0])
else:  # user sent us a wildcard, need to use glob to find files
    files = glob.glob(args.basepath + args.input_pattern[0])


# In[ ]:


# ------ work on files
for fname in files:
    
    fname = fname.strip()
    newfname_class = "%s/%s_class.png" % (OUTPUT_DIR, os.path.basename(fname)[0:fname.rfind('.')])

    print(f"working on file: \t {fname}")
    print(f"saving to : \t {newfname_class}")

    if not args.force and os.path.exists(newfname_class):
        print("Skipping as output file exists")
        continue
        
    cv2.imwrite(newfname_class, np.zeros(shape=(1, 1)))                                            
    
    xml_dir = fname[0:fname.rfind(os.path.sep)]
    xml_fname = xml_dir + os.path.sep + os.path.basename(fname)[0:os.path.basename(fname).rfind('.')] + '.xml'

    img = wsi.wsi(fname,xml_fname)
    
    stride_size = int(base_stride_size * (args.resolution/img["mpp"]))
    
    if(args.annotation.lower() == 'wsi'):
        img_dims = [0,0,w["img_dims"][0],w["img_dims"][1]]
    else:
        img_dims = img.get_dimensions_of_annotation(args.color,args.annotation)
    
    if img_dims:
    
        x_start = int(img_dims[0])
        y_start = int(img_dims[1])
        w_orig = int(img_dims[2])
        h_orig = int(img_dims[3])


        w = int(w_orig + (patch_size - (w_orig % patch_size)))
        h = int(h_orig + (patch_size - (h_orig % patch_size)))

        x_points = range(x_start-stride_size//2,x_start+w+stride_size//2,stride_size)
        y_points = range(y_start-stride_size//2,y_start+h+stride_size//2,stride_size)

        grid_points = [(x,y) for x in x_points for y in y_points]
        points_split = divide_batch(grid_points,batch_size)

        #in case we have a large network, lets cut the list of tiles into batches
        output = np.zeros((len(grid_points),checkpoint["n_classes"],patch_size,patch_size))
        for i,batch_points in enumerate(points_split):

            batch_arr = np.array([img.get_tile(args.resolution,coords,(patch_size,patch_size)) for coords in batch_points])
#             print(np.shape(arr_out))
#             arr_out = arr_out.reshape(-1,patch_size,patch_size,3)

            arr_out_gpu = torch.from_numpy(batch_arr.transpose(0, 3, 1, 2) / 255).type('torch.FloatTensor').to(device)

            # ---- get results
            output_batch = model(arr_out_gpu)

            # --- pull from GPU and append to rest of output 
            output_batch = output_batch.detach().cpu().numpy()

            output[((i+1)*batch_size - batch_size):((i+1)*batch_size),:,:,:] = output_batch
#             output = np.append(output,output_batch,axis=0)


        output = output.transpose((0, 3, 2, 1))

        #turn from a single list into a matrix of tiles
        output = output.reshape(len(x_points),len(y_points),patch_size,patch_size,output.shape[3])

        #remove the padding from each tile, we only keep the center
        output=output[:,:,base_stride_size//2:-base_stride_size//2,base_stride_size//2:-base_stride_size//2,:]

        #turn all the tiles into an image
        output=np.concatenate(np.concatenate(output,1),1)

        #incase there was extra padding to get a multiple of patch size, remove that as well
        mask = img.get_annotated_region(args.resolution,args.color,args.annotation,return_img=False)
        mask = mask[1]

        output = output.transpose((1,0,2))
        output = output[0:mask.shape[0], 0:mask.shape[1], :] #remove paddind, crop back
        output = output.argmax(axis=2) * (256 / (output.shape[-1] - 1) - 1)
        output = cv2.bitwise_and(output,output,mask=np.uint8(mask))
        output = np.uint8((output>0) + (mask==0))*255
        
        # --- save output

        # cv2.imwrite(newfname_class, (output.argmax(axis=2) * (256 / (output.shape[-1] - 1) - 1)).astype(np.uint8))
        cv2.imwrite(newfname_class, output)
        
    else:
        print('No annotation of color')

