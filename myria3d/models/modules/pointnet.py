import torch
import torch.nn as nn

import numpy as np
from sklearn.neighbors import NearestNeighbors
from myria3d.utils import utils  # noqa

class PointNet(nn.Module):
  """
  PointNet network for semantic segmentation
  """

  def __init__(
      self,
      num_classes: int,
      num_features: int,
      subsample: int = 512,
      MLP_1: list = [32,64],
      MLP_2: list = [64,128,256],
      MLP_3: list = [128,64,32],
      ):
    """
    initialization function
    num_classes : int =  the number of class
    num_features : int = number of input feature
    MLP_1, MLP_2 and MLP_3 : int list = width of the layers of
    multi-layer perceptrons. For example MLP_1 = [32, 64] or [16, 64, 128]
    cuda : int = if 0 run on CPU (slow but easy to debug), if 1 on GPU
    """

    super(PointNet, self).__init__() #necessary for all classes extending the module class
    self.subsample = subsample
    self.num_features = num_features + 3 #pos

    #since we don't know the number of layers in the MLPs, we need to use loops
    #to create the correct number of layers

    m1 = MLP_1[-1] #size of the first embeding F1
    m2 = MLP_2[-1] #size of the second embeding F2

    #MLP_1: input [num_features x n] -> f1 [m1 x n]
    modules = []
    for i in range(len(MLP_1)): #loop over the layer of MLP1
      #note: for the first layer, the first in_channels is feature_size
      modules.append(
          nn.Conv1d(in_channels=MLP_1[i-1] if i>0 else self.num_features, #to handle i=0
                    out_channels=MLP_1[i], kernel_size=1))
      modules.append(nn.BatchNorm1d(MLP_1[i]))
      modules.append(nn.ReLU(True))
    #this transform the list of layers into a callable module
    self.MLP_1 = nn.Sequential(*modules)

    #MLP_2: f1 [m1 x n] -> f2 [m2 x n]
    modules = []
    for i in range(len(MLP_2)):
      modules.append(nn.Conv1d(in_channels=MLP_2[i-1] if i>0 else m1,
                               out_channels=MLP_2[i], kernel_size=1))
      modules.append(nn.BatchNorm1d(MLP_2[i]))
      modules.append(nn.ReLU(True))
    self.MLP_2 = nn.Sequential(*modules)

    #MLP_3: f1 [(m1 + m2) x n] -> output [k x n]
    modules = []
    for i in range(len(MLP_3)):
      modules.append(nn.Conv1d(in_channels=MLP_3[i-1] if i>0 else m1+m2,
                               out_channels=MLP_3[i], kernel_size=1))
      modules.append(nn.BatchNorm1d(MLP_3[i]))
      modules.append(nn.ReLU(True))

    #note: the last layer do not have normalization nor activation
    modules.append(nn.Conv1d(MLP_3[-1], num_classes, 1))

    self.MLP_3 = nn.Sequential(*modules)

  def forward(self, x, pos, batch, ptr):
    """
    the forward function producing the embeddings for each point of 'input'
    input : [n_batch, num_features, n_points] float array = input features
    output : [n_batch, num_classes, n_points] float array = point class logits
    """
    n_batch = ptr.size(0) - 1
    n_points = len(pos)
    input_all=torch.cat((pos,x), axis=1)
    input=torch.Tensor(n_batch, self.num_features, self.subsample)
    out=torch.Tensor(n_batch, self.num_features+1, n_points)

    for i_batch in range(n_batch):
      b_idx = np.where(batch.cpu()==i_batch)
      full_cloud = input_all[b_idx]
      n_full = full_cloud.shape[0]
      selected_points = np.random.choice(n_full, self.subsample)
      input_batch = full_cloud[selected_points]
      input[i_batch,:,:] = input_batch.T

    input = input.to(ptr.device)

    #embed points, equation (1)
    b1_out = self.MLP_1(input)

    #second point embeddings equation (2)
    b2_out = self.MLP_2(b1_out)

    #maxpool, equation 3
    G = torch.max(b2_out,2,keepdim=True)[0]

    #concatenate f1 and G
    Gf1 = torch.cat((G.repeat(1,1,self.subsample), b1_out),1)

    #equation(4)
    pred = self.MLP_3(Gf1)
    pred = pred.permute(0,2,1).flatten(0,1)
    
    pos_sampled = input[:,:3,:].permute(0,2,1).flatten(0,1)

    knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit( \
                            pos_sampled.cpu())
    
    _, closest_point = knn.kneighbors(pos.cpu())

    #remove uneeded dimension (we only took one neighbor)
    closest_point = closest_point.squeeze()

    out = pred[closest_point,:]

    return out
  
