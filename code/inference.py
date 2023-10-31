from argparse import Namespace
import os
import sys
import pprint
import base64
from io import BytesIO
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

os.chdir("/opt/ml/code/SAM")
sys.path.append(".")
sys.path.append("..")

from datasets.augmentations import AgeTransformer
from utils.common import tensor2im
from models.psp import pSp
import dlib
from scripts.align_all_parallel import align_face

trsfms = transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

## Helper functions ##

def run_alignment(image_path):
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor) 
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image

def run_on_batch(inputs, net):
    result_batch = net(inputs.to("cuda").float(), randomize_noise=False, resize=False)
    return result_batch

## Inference functions ##

def model_fn(model_dir):
    print(f"inside model_fn, model_dir= {model_dir}")
    ckpt = torch.load(os.path.join(model_dir, "model.pt"),map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = os.path.join(model_dir, "model.pt")
    opts = Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    return net


def input_fn(request_body, request_content_type):
    '''WIP: Expecting image/jpeg content type'''
    if request_content_type != 'image/jpeg':
            raise Exception(f'Requested unsupported ContentType in request content_type {request_content_type}')
    print('input_fn: Deserializing input data')
    ## Size of the image is predecided, comes sized from the client.
    orig_img = Image.open(BytesIO(base64.b64decode(request_body))).convert("RGB")
    print('input_fn: saving to a new image file')
    orig_img.save('input_image.jpg')
    print('input_fn: running alignment for the input image')
    aligned_image = run_alignment('input_image.jpg')
    print('input_fn: resizing the image to 256 x 256')
    aligned_image.resize((256,256))
    print('input_fn: applying transformation to the resized image')
    final_input_image = trsfms(aligned_image)
    print('input_fn: exiting input_fn')
    return final_input_image

#### Lets comment each of the custom implimentations - one at a time ####
#def predict_fn(input_data, model):
#    final_input_image = input_data
#    print('Generating prediction based on input')
#    # we'll run the image on multiple target ages 
#    target_ages = [10, 50, 90]
#    age_transformers = [AgeTransformer(target_age=age) for age in target_ages]
#    # for each age transformed age, we'll concatenate the results to display them side-by-side
#    results = np.array(final_input_image.resize((1024, 1024)))
#    for age_transformer in age_transformers:
#        print(f"Running on target age: {age_transformer.target_age}")
#        with torch.no_grad():
#            input_image_age = [age_transformer(final_input_image.cpu()).to('cuda')]
#            input_image_age = torch.stack(input_image_age)
#            result_tensor = run_on_batch(input_image_age, model)[0]
#            result_image = tensor2im(result_tensor)
#            results = np.concatenate([results, result_image], axis=1)
#    return results

## output_fn() need not be implemented as it accepts NPY (application/x-npy) as a
## response content type. We will use the capabilities of the client side to parse
## the content and display the pictures.
