from collections import OrderedDict
import torch
from bagnet_utils import plot_heatmap, generate_heatmap_pytorch
import mbag
import argparse
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
    elif patch_size == 17:
        model = mbag.bagnet17_18()
    elif patch_size == 33:
        model = mbag.bagnet33_18()
    else:
        raise ValueError("Invalid patch_size")
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
def load_sample(path_name, size = (400,400)):
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

def save_figure(path_to_figure:str,original_image, heatmap,title="bagnet"):
    fig = plt.figure(figsize=(8, 4))
    fig.suptitle(title)
    original_image = original_image[0].transpose([1,2,0])
    ax = plt.subplot(121)
    ax.set_title('original')
    plt.imshow(original_image[:,:,0] / 255.)
    plt.axis('off')

    ax = plt.subplot(122)
    ax.set_title('heatmap')
    plot_heatmap(heatmap, original_image, ax, dilation=0.5, percentile=99, alpha=.25)
    plt.axis('off')
    fig.savefig(path_to_figure)

def load_csv_data(path_to_csv):
    df = pd.read_csv(path_to_csv, sep=';')
    outcomes = []
    abnormalities = list(df['AbnormalityType'])
    img_names = list(df['ImgName'])
    for model in ('Groundtruth','bagnet9','bagnet17','bagnet33'):
        outcomes.append(list(df[model]))
    return img_names, abnormalities, outcomes

def analyze_heatmaps(path_to_csv,path_to_data, path_to_figure, image_size=800):
    groundtruth_dict = {0:"benign",1:"malignant"}
    patch_dict = {0:9,1:17,2:33}

    img_names, abnormalities, outcomes = load_csv_data(path_to_csv)
    model_idx=0
    for model_predictions in outcomes[:][1:]:
        patch_size = patch_dict[model_idx]
        model_idx += 1
        path_to_model = get_model_path_name(patch_size)
        model = load_model(path_to_model, patch_size)
        device = torch.device("cuda")
        model.to(device)
        for img_name, groundtruth, prediction, abnormality in zip(img_names, outcomes[:][0], model_predictions, abnormalities):
            path_to_sample = path_to_data+img_name+'.png'
            figure_name = "heatmap_"+img_name+"_bagnet"+str(patch_size)+".png"
            figure_title = str(abnormality)+" "+str(groundtruth_dict[groundtruth]) +" BagNet"+str(patch_size)+" predicted: "+groundtruth_dict[prediction]
            #load model
            model = load_model(path_to_model, patch_size)
            device = torch.device("cuda")
            model.to(device)
            path_to_figure = path_to_figure + figure_name
            sample = load_sample(path_to_sample, size=(image_size,image_size))
            heatmap = generate_heatmap_pytorch(model, sample, groundtruth, patch_size)
            save_figure(path_to_figure, sample, heatmap, figure_title)
        # free up memory
        del model

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="1 if malignant, else 0 ", type=str, default="/mbag-REDI/mbag/heatmap_pictures.csv")
    args = parser.parse_args()
    filename = args.filename
    base_path = os.path.abspath(os.getcwd())
    path_to_csv = base_path + filename
    path_to_data = "deepstore/datasets/dmb/Biomedical/cbis-ddsm/processed/"
    path_to_figure = base_path +"/heatmaps/"
    analyze_heatmaps(path_to_csv,path_to_data,path_to_figure)
    # base_path = os.path.abspath(os.getcwd())
   
    # figure_title = "Heatmap of BagNet_"+str(patch_size)

    # path_to_model = get_model_path_name(patch_size)
 
    
    # path_to_figure = base_path+"/mbag-REDI/multiview_mammogram/results/"+figure_name
    
    
 