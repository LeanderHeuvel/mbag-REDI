from collections import OrderedDict
import torch
from bagnet_utils import plot_heatmap, generate_heatmap_pytorch
import mbag
import argparse
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


## remove module. namespace
def remove_module(state_dict):
  new_state_dict = OrderedDict()
  for k, v in state_dict['state_dict'].items():
    name = k.replace("module.", "")
    new_state_dict[name]=v
  return new_state_dict
def load_model(model_path, patch_size=33):
    '''
    Loads pretrained BagNet model with state dict
    Params:
    path of model: String

    Returns:
    model loaded with state_dict

    '''
    if patch_size == 9:
        model = mbag.bagnet9_18()
    if patch_size == 17:
        model = mbag.bagnet17_18()
    if patch_size == 33:
        model = mbag.bagnet33_18()
    else:
        raise ValueError("Invalid path_size")
    checkpoint = torch.load(model_path)
    checkpoint = remove_module(checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model
def get_model_path_name(patch_size):
    '''
    returns the path to the corresponding stored model for the image patch size
    Params:
    Image_patch_size: Int
    Returns:
    pathname: String
    '''
    modality='MG'
    base_path = os.path.abspath(os.getcwd())+"/mbag-REDI"
    file_name="sota_singlepipeline_MILpoolingavg_softmax_variableview_batch15_pe10_oldresult_run2_bagnet_"+str(patch_size)+'_'+modality #name of the output files that will be created
    path_to_model = base_path+"/multiview_mammogram/models/"+file_name+".tar" #name of the folder where to save the models
    
    return path_to_model
def load_sample(path_name, size = (800,800)):
    '''
    returns img array of image in 1 x width x height

    '''
    x, y = size
    img = Image.open(path_name).convert('L')
    img = img.resize((x,y))
    img = np.asarray(img)
    img = np.expand_dims(img,axis=0)
    img = np.expand_dims(img,axis=0)
    return img

def save_figure(path_to_figure:str,original_image, heatmap):
    fig = plt.figure(figsize=(8, 4))
    original_image = original_image[0].transpose([1,2,0])
    ax = plt.subplot(121)
    ax.set_title('original')
    plt.imshow(original_image / 255.)
    plt.axis('off')

    ax = plt.subplot(122)
    ax.set_title('heatmap')
    plot_heatmap(heatmap, original_image, ax, dilation=0.5, percentile=99, alpha=.25)
    plt.axis('off')
    fig.savefig(path_to_figure)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", help="size of square image with height = width, ", type=int, default=400)
    parser.add_argument("--patch_size", help="image patch size of the model, ", type=int, default=9)
    parser.add_argument("--malignant", help="1 if malignant, else 0 ", type=int, default=0)
    args = parser.parse_args()
    image_size = args.image_size
    patch_size = args.patch_size
    malignant = args.malignant

    base_path = os.path.abspath(os.getcwd())
    path_to_sample = base_path+"/Calc-Test_P_00038_LEFT_CC_1-1.png"

    path_to_model = get_model_path_name(patch_size)
    label = malignant
    figure_name = "heatmap_bagnet_"+str(patch_size)+".png"
    path_to_figure = base_path+"/mbag-REDI/multiview_mammogram/results/"+figure_name
    
    print("Loading model...")
    model = load_model(path_to_model)
    device = torch.device("cuda")
    model.to(device)
    print("Loading sample")
    sample = load_sample(path_to_sample)
    print("Generating heatmap...")
    heatmap = generate_heatmap_pytorch(model, sample, label, patch_size)
    print("Saving figure...")
    save_figure(path_to_figure, sample, heatmap)
    print("Done!")
