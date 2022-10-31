# -*- coding: utf-8 -*-
""" Deletion and insertion metric are introduced by 
Petsiuk, Vitali, Abir Das, and Kate Saenko. 
"RISE: Randomized Input Sampling for Explanation of Black-box Models."
We use the authors' implenmentation.

Reference: https://github.com/eclique/RISE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import scipy.io
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

import torch
import load_db

# Constants
HW = 224 * 224  # Image area
n_classes = 1000  # Number of classes in ImageNet


def gkern(klen, nsig):
  """Returns a Gaussian kernel array.
  Convolution with it results in image blurring."""
  # create nxn zeros
  inp = np.zeros((klen, klen))
  # set element at the middle to one, a dirac delta
  inp[klen//2, klen//2] = 1
  # gaussian-smooth the dirac, resulting in a gaussian filter mask
  k = gaussian_filter(inp, nsig)
  kern = np.zeros((3, 3, klen, klen))
  kern[0, 0] = k
  kern[1, 1] = k
  kern[2, 2] = k
  return torch.from_numpy(kern.astype('float32'))

def auc(arr):
  """Returns normalized Area Under Curve of the array."""
  return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


class CausalMetric():

  def __init__(self, model, mode, step, substrate_fn):
    """Create deletion/insertion metric instance.

    Args:
      model (nn.Module): Black-box model being explained.
      mode (str): 'del' or 'ins'.
      step (int): number of pixels modified per one iteration.
      substrate_fn (func): a mapping from old pixels to new pixels.
    """
    assert mode in ['del', 'ins']
    self.model = model
    self.mode = mode
    self.step = step
    self.substrate_fn = substrate_fn

  def single_run(self, img_tensor, explanation, verbose=0, save_to=None):
    """Run metric on one image-saliency pair.

    Args:
      img_tensor (Tensor): normalized image tensor.
      explanation (np.ndarray): saliency map.
      verbose (int): in [0, 1, 2].
        0 - return list of scores.
        1 - also plot final step.
        2 - also plot every step and print 2 top classes.
      save_to (str): directory to save every step plots to.

    Return:
      scores (nd.array): Array containing scores at every step.
    """
    pred = self.model(img_tensor.cuda())
    top, c = torch.max(pred, 1)
    c = c.cpu().numpy()[0]
    n_steps = (HW + self.step - 1) // self.step

    if self.mode == 'del':
      title = 'Deletion game'
      ylabel = 'Pixels deleted'
      start = img_tensor.clone()
      finish = self.substrate_fn(img_tensor)
    elif self.mode == 'ins':
      title = 'Insertion game'
      ylabel = 'Pixels inserted'
      start = self.substrate_fn(img_tensor)
      finish = img_tensor.clone()

    scores = np.empty(n_steps + 1)
    # Coordinates of pixels in order of decreasing saliency
    salient_order = np.flip(np.argsort(explanation.reshape(-1, HW).numpy(), axis=1), axis=-1)
    for i in range(n_steps+1):
      pred = self.model(start.cuda())
      pr, cl = torch.topk(pred, 2)
      scores[i] = pred[0, c]
      # Render image if verbose, if it's the last step or if save is required.
      if verbose == 2 or (verbose == 1 and i == n_steps) or save_to:
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.title('{} {:.1f}%, P={:.4f}'.format(ylabel, 100 * i / n_steps, scores[i]))
        plt.axis('off')

        plt.subplot(122)
        plt.plot(np.arange(i+1) / n_steps, scores[:i+1])
        plt.xlim(-0.1, 1.1)
        plt.ylim(0, 1.05)
        plt.fill_between(np.arange(i+1) / n_steps, 0, scores[:i+1], alpha=0.4)
        plt.title(title)
        plt.xlabel(ylabel)
        #plt.ylabel(get_class_name(c))
        if save_to:
          plt.savefig(save_to + '/{:03d}.png'.format(i))
          plt.close()
        else:
          plt.show()
      if i < n_steps:
        coords = salient_order[:, self.step * i:self.step * (i + 1)]
        start.cpu().numpy().reshape(1, 3, HW)[0, :, coords] = finish.cpu().numpy().reshape(1, 3, HW)[0, :, coords]
    return scores

  def evaluate(self, img_batch, exp_batch, batch_size):
    """Efficiently evaluate big batch of images.

    Args:
      img_batch (Tensor): batch of images.
      exp_batch (np.ndarray): batch of explanations.
      batch_size (int): number of images for one small batch.

    Returns:
      scores (nd.array): Array containing scores at every step for every image.
    """
    n_samples = img_batch.shape[0]
    predictions = torch.FloatTensor(n_samples, n_classes)
    assert n_samples % batch_size == 0
    for i in range(n_samples // batch_size):
      preds = self.model(img_batch[i*batch_size:(i+1)*batch_size].cuda()).cpu()
      predictions[i*batch_size:(i+1)*batch_size] = preds
    top = np.argmax(predictions.detach(), -1)
    n_steps = (HW + self.step - 1) // self.step
    scores = np.empty((n_steps + 1, n_samples))
    salient_order = np.flip(np.argsort(exp_batch.reshape(-1, HW), axis=1).numpy(), axis=-1)
    r = np.arange(n_samples).reshape(n_samples, 1)

    substrate = torch.zeros_like(img_batch)
    for j in range(n_samples // batch_size):
      substrate[j*batch_size:(j+1)*batch_size] = self.substrate_fn(img_batch[j*batch_size:(j+1)*batch_size])

    if self.mode == 'del':
      caption = 'Deleting  '
      start = img_batch.clone()
      finish = substrate
    elif self.mode == 'ins':
      caption = 'Inserting '
      start = substrate
      finish = img_batch.clone()

    # While not all pixels are changed
    for i in range(n_steps+1):
      # Iterate over batches
      for j in range(n_samples // batch_size):
        # Compute new scores
        preds = self.model(start[j*batch_size:(j+1)*batch_size].cuda()).detach()
        preds = preds.cpu().numpy()[range(batch_size), top[j*batch_size:(j+1)*batch_size]]
        scores[i, j*batch_size:(j+1)*batch_size] = preds
      # Change specified number of most salient pixels to substrate pixels
      coords = salient_order[:, self.step * i:self.step * (i + 1)]
      start.cpu().numpy().reshape(n_samples, 3, HW)[r, :, coords] = finish.cpu().numpy().reshape(n_samples, 3, HW)[r, :, coords]
    return scores


def delins_test_with_heatmap(
  inv_dir, 
  pretrained_model, 
  batch_sz,
  imagenet_val_path,
  val_gt_path,
  val_gt_txt_path,
  device,
  ):
  """ Compute insertion/deletion metric from heatmaps.

  Args:
    inv_dir (str): Path of inverse results
    pretrained_model (torch.nn): Pretrained model to be tested
    batch_sz (int): Batch size for computing both metrics
    imagenet_val_path (str): Path for Validation split images.
    val_gt_path (str): Path for the ground truth text file for val split
    val_gt_txt_path (str): Path for the text description of GT of val split
    device (str): Device for torch
  """
  # Define model, db, and device
  model = torch.nn.Sequential(pretrained_model, torch.nn.Softmax(dim=1))
  model = model.to(device)

  # Get LoadDB instance
  #imagenet = load_db.ImageNet(val_dir=imagenet_val_path, device=device)
  imagenet = load_db.ImageNet(
    val_dir=imagenet_val_path,
    val_gt_path=val_gt_path,
    val_gt_txt_path=val_gt_txt_path,
    device=device,
  )

  # Get deletion metric object
  del_metric = CausalMetric(model, 'del', 224, substrate_fn=torch.zeros_like)

  # Get insertion metric object
  klen = 11
  ksig = 5
  kern = gkern(klen, ksig)
  blur = lambda x: torch.nn.functional.conv2d(x, kern, padding=klen // 2)
  ins_metric = CausalMetric(model,
                            'ins', 224, substrate_fn=blur)

  # Get MAT list
  mat_file_list = sorted(glob.glob(os.path.join(inv_dir, '*.mat')))

  # Iterate over MAT files
  mat_file_batch_list = [mat_file_list[i:i + batch_sz] 
                         for i in range(0, len(mat_file_list), batch_sz)]

  data_idxs = [] 
  max_probs = []
  predictions = []
  labels = []
  del_aucs = []
  ins_aucs = []
  del_scores = []
  ins_scores = []

  for batch_idx, mat_file_batch in enumerate(mat_file_batch_list):
    
    print('Evaluation of {}th batch.'.format(batch_idx))

    sal_maps = []
    input_images = []

    for file_idx, mat_file in enumerate(mat_file_batch):

      data_idx = imagenet.get_validation_data_idx(mat_file)
      input_image, input_label, _ = imagenet.validation_data(data_idx)
      input_image = imagenet.pre_processing(input_image, 
        with_normalization=True
      )

      # Calculate saliency map for normal inv
      mat_contents = scipy.io.loadmat(mat_file)
      mat_keys = list(mat_contents.keys())
      sal_map = mat_contents[mat_keys[3]]
      sal_map = torch.tensor(sal_map / sal_map.max())
      sal_map = sal_map.unsqueeze(0).unsqueeze(0)
      
      data_idxs.append(data_idx)
      labels.append(input_label)      
      if file_idx == 0:
        sal_maps = sal_map
        input_images = input_image
      else:
        sal_maps = torch.cat([sal_maps, sal_map], dim=0)
        input_images = torch.cat([input_images, input_image], dim=0)
      
    # Calculate deletion metric
    torch.cuda.empty_cache()
    del_score = del_metric.evaluate(
      img_batch=input_images.cpu(),
      exp_batch=sal_maps,
      batch_size=sal_maps.shape[0])

    # Calculate insertion metric
    torch.cuda.empty_cache()
    ins_score = ins_metric.evaluate(
      img_batch=input_images.cpu(),
      exp_batch=sal_maps,
      batch_size=sal_maps.shape[0])

    # Get predictions and max_probs
    with torch.no_grad():
      logit = model(input_images)
      max_prob, prediction = torch.max(logit, dim=1)
      torch.cuda.empty_cache()
    
    # Save data
    if batch_idx == 0:
      del_scores = del_score
      ins_scores = ins_score
      predictions = prediction
      max_probs = max_prob
    else:
      del_scores = np.concatenate((del_scores, del_score), axis=1)  
      ins_scores = np.concatenate((ins_scores, ins_score), axis=1)  
      predictions = torch.cat([predictions, prediction], dim=0)
      max_probs = torch.cat([max_probs, max_prob], dim=0)

  n_aucs = del_scores.shape[1]
  del_aucs = np.zeros([n_aucs])  
  ins_aucs = np.zeros([n_aucs])  

  total_del_auc = 0
  total_ins_auc = 0
  for i in range(n_aucs):
    del_aucs[i] = auc(del_scores[:, i])
    ins_aucs[i] = auc(ins_scores[:, i])
    total_del_auc += del_aucs[i]
    total_ins_auc += ins_aucs[i]
  mean_del_auc = total_del_auc / n_aucs
  mean_ins_auc = total_ins_auc / n_aucs

  result = {'data_idxs': data_idxs, 
            'deletion_scores': np.array(del_aucs),
            'insertion_scores': np.array(ins_aucs), 
            'prediction_probabilties': max_probs, 
            'predictions': predictions, 
            'labels': labels}
            
  print('del_auc = {:.7f}, ins_auc = {:.7f}'.format(mean_del_auc, mean_ins_auc))
  
  return result