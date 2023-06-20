# %% [markdown]
# ## Installing Tools

# %%
import sys, os, distutils.core
# Properly install detectron2.
#which python3
#python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# %%
#!git clone https://github.com/facebookresearch/detectron2.git

# %% [markdown]
# ## Importing Tools

# %%
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import torch

import tensorflow as tf
from keras.models import Model, load_model

# import some common libraries
import numpy as np
import os, json, cv2, random
import glob
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize

import PIL


# %%
#!nvcc --version
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

print("GPU", torch.cuda.is_available())

print('---------TENSORFLOW-----------------')
print(tf.__version__)
print("Available: ", tf.config.list_physical_devices())

# %% [markdown]
# ## Get SyRIP Data

# %%
#!wget -r -np -A "train0*.jpg" https://coe.northeastern.edu/Research/AClab/SyRIP/images/train_infant/ --no-host-directories -Q

# %%
## Use only for direct terminal access
#!pip install colab-xterm
#%load_ext colabxterm
#%xterm

# %%
#!echo 'current working directory :' && pwd
#%cd /home/cescollino/Documents/ETS/MAITRISE/Dataset/Baby-Dataset-master/dataset/full/
#Research/AClab/SyRIP/images/train_infant/
#!echo "Real Images" && ls | head -n 5 && ls | tail -n 5 && echo 'current working directory :' && pwd
# 500 images total

# %% [markdown]
# # Data preprocessing
# 

# %% [markdown]
# ## Image Histograms

# %%
def describeimage(image, bins=256):
    plt.rcParams["figure.figsize"] = (20,5)
    plt.imshow(image)
    #plt.set_cmap('gist_gray')
    #plt.colorbar()
    plt.show()
    #print("The shape of this image is",np.shape(image))
    #print("The minimum value of this image is",np.min(image))
    #print("The maximum value of this image is",np.max(image))
    #print("The mean value of this image is",np.mean(image))
    
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.rcParams["figure.figsize"] = (20,5)
    plt.figure()
    plt.title("'Flattened' Color Histogram")
    plt.xlabel("Bins Excluding 0 (Padding)")
    plt.ylabel("# of Pixels")
    # loop over the image channels
    for (chan, color) in zip(chans, colors):
      # create a histogram for the current channel and plot it
      hist = cv2.calcHist([chan], [0], None, [256], [1, 256])
      plt.plot(hist, color=color)
      plt.xlim([0, 256])
    #plt.xlim(0,1)
    plt.show()

# %% [markdown]
# ## Padding

# %%
#realbabies = glob.glob("./train0*.jpg")
realbabies = glob.glob("/home/cescollino/Documents/ETS/MAITRISE/Dataset/Baby-Dataset-master/dataset/full/*.jpg")
print("dataset lenght:", len(realbabies))
IMG_WIDTH = 512
print("IMG_WIDTH",IMG_WIDTH)

IMG_CHANNELS = 3
print("IMG_CHANNELS",IMG_CHANNELS)

X_BB = np.zeros((len(realbabies), IMG_WIDTH, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

def padding(im, desired_size:int):
  
  old_size = im.shape[:2] # old_size is in (height, width) format
  #print(old_size)
  ratio = float(desired_size)/max(old_size)
  new_size = tuple([int(x*ratio) for x in old_size])
  im = cv2.resize(im, (new_size[1], new_size[0]))
  delta_w = desired_size - new_size[1]
  delta_h = desired_size - new_size[0]
  top, bottom = delta_h//2, delta_h-(delta_h//2)
  left, right = delta_w//2, delta_w-(delta_w//2)
  color = [0, 0, 0]
  new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
  
  return new_im


for n, id_ in tqdm(enumerate(realbabies), total=len(realbabies)):
    img = cv2.imread(id_)[:,:,:IMG_CHANNELS]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if np.shape(img)[2] != 3: #verify channels
      print(np.shape(img)[2],n)
    #print("b4 padding",np.shape(img))
    img = padding(img,IMG_WIDTH)
    #print("after padding",np.shape(img))
    img = resize(img, (IMG_WIDTH, IMG_WIDTH), mode='constant', preserve_range=True)
    X_BB[n] = img


# %% [markdown]
# ## Example images from test np array

# %%
ix = random.randint(0, np.shape(X_BB)[0])
describeimage(X_BB[ix])


# %% [markdown]
# ## U-NET skin segmentation

# %%
## Intersection-over-Union (IoU) metric, can be tracked instead of the accuracy during training

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# %%
#!pwd
model = load_model('/home/cescollino/Workspace/abd-skin-segmentation/UNET/your_model_name.h5', custom_objects={'mean_iou': mean_iou})

# %% [markdown]
# ## MRCNN Person Segm
# 
# {0: u'__background__',
#  1: u'**person**',
#  2: u'bicycle',
#  ...
#  79: u'hair drier',
#  80: u'toothbrush'}

# %% [markdown]
# ## Only Segmenting Person Class
# ![image.png](attachment:1aba4d61-190d-4b52-9e13-587374177f25.png)
# ![image.png](attachment:c8b73c86-bf06-439f-9be9-902d6b24cceb.png)

# %%
def onlykeeppersonclass(outputs, cuda_available:bool):
  cls = outputs['instances'].pred_classes
  scores = outputs["instances"].scores
  masks = outputs['instances'].pred_masks
  boxes = outputs['instances'].pred_boxes
  print(type(outputs['instances']))
  print(boxes.tensor.size())
  raise(IndexError('wait!'))
  # remove all other classes which are not person(index:0)
  indx_to_remove = (cls != 0).nonzero().flatten().tolist()
  
  # delete corresponding arrays
  cls = np.delete(cls.cpu().numpy(), indx_to_remove)
  scores = np.delete(scores.cpu().numpy(), indx_to_remove)
  masks = np.delete(masks.cpu().numpy(), indx_to_remove, axis=0)
  boxes = torch.cat((boxes.tensor[:],boxes.tensor[:]), axis = 1)
  #t = torch.cat((t[:,:3], t[:,4:]), axis = 1)))
  
  if cuda_available:
      
    # convert back to tensor and move to cuda
    cls = torch.tensor(cls).to('cuda:0')
    scores = torch.tensor(scores).to('cuda:0')
    masks = torch.tensor(masks).to('cuda:0')
    boxes = torch.tensor(boxes).to('cuda:0')
  else:
    cls = torch.tensor(cls)
    scores = torch.tensor(scores)
    masks = torch.tensor(masks)
    boxes = torch.tensor(boxes)

  # if not interested in boxes
  #outputs['instances'].remove('pred_boxes')

  # create new instance obj and set its fields
  obj = detectron2.structures.Instances(image_size=(IMG_WIDTH, IMG_WIDTH))
  obj.set('pred_classes', cls)
  obj.set('scores', scores)
  obj.set('pred_masks', masks)
  obj.set('pred_boxes', boxes)
  
  return obj

# %% [markdown]
# ## Instance Segmentation Inference

# %%
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cuda_available = torch.cuda.is_available()
if cuda_available == False:
  print("GPU not available")
  cfg.MODEL.DEVICE='cpu'

predictor = DefaultPredictor(cfg)

for i in tqdm(X_BB, total = len(X_BB)):
  outputs = predictor(i)
  
  #outputs = onlykeeppersonclass(outputs, cuda_available)
  
  v = Visualizer(i[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
  
  out = v.draw_instance_predictions(outputs['instances'].to('cpu'))
  
  img = out.get_image()
  
  
  #print(outputs["instances"])
  
  for n,bbox in enumerate(iter(list(outputs["instances"].pred_boxes))):
    bbox = bbox.detach().cpu().numpy().astype(int)
    print(bbox)
    crop_img = out.get_image()[:, :, ::-1]
    crop_img = i[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    
    ##Segment Person
    crop_mask  = outputs["instances"].pred_masks.detach().cpu().numpy()
    print('crop_mask shape::',np.shape(crop_mask))
    crop_mask = np.moveaxis(crop_mask,0,2)
    crop_mask = crop_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    crop_mask = np.stack([crop_mask,crop_mask,crop_mask],axis=2)
    crop_mask = np.squeeze(crop_mask)
    crop_mask = np.array(crop_mask, dtype=np.uint8)
    
    crop_result = crop_img*crop_mask
    
    ## Segment Skin
    print(np.shape(crop_result))
    resized = padding(crop_result, np.max(np.shape(crop_result)))
    resized = cv2.resize(resized,(128,128))
    print(np.shape(resized))
    plt.imshow(resized)
    plt.show()
  
    skin = model.predict(np.stack(([resized,resized]))) ##workaround, messy
    skin = skin[0]
    plt.imshow(skin)
    plt.show()
    plt.imshow(crop_result)
    plt.show()
  
  
  plt.imshow(out.get_image()[:, :, ::-1])
  #plt.set_cmap('gist_gray')
  #plt.colorbar()
  plt.show()
    

# %% [markdown]
# ## Inference Panoptic Segmentation Data

# %%
# Inference with a keypoint detection model
cfg = get_cfg()   # get a fresh new config
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
predictor = DefaultPredictor(cfg)

for i in X_BB:
  panoptic_seg, segments_info = predictor(i)["panoptic_seg"]
  panoptic_seg[panoptic_seg < 0] = 0
  v = Visualizer(i[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
  out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
  #image = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
  plt.imshow(out.get_image()[:, :, ::-1])
  plt.show()

# %% [markdown]
# ## Inference with a keypoint detection model

# %%
# Inference with a keypoint detection model
cfg = get_cfg()   # get a fresh new config
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
for i in X_BB:
  outputs = predictor(i)
    
  v = Visualizer(i[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  describeimage(out.get_image()[:, :, ::-1])

# %% [markdown]
# ## Inference with DensePose

# %%
#!pip install -v git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose


# %%
#!wget https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_s1x/165712116/model_final_844d15.pkl 

# %%
#!python /content/detectron2/projects/DensePose/apply_net.py show /content/detectron2/projects/DensePose/configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml ./model_final_844d15.pkl input.jpg dp_contour,bbox --output image_densepose_contour.png


# %%
#a= cv2.imread("image_densepose_contour.0001.png")
#cv2_imshow(a)


