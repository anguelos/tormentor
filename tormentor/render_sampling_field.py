import torch
import cv2;
import kornia as K

img=torch.from_numpy(cv2.imread("/home/anguelos/Desktop/calibrationimage.jpg").transpose([2,0,1])).unsqueeze(0)[:,[2,1,0],:,:]
m = K.utils.create_meshgrid(img.size(2),img.size(3), normalized_coordinates=False)
X = m[:, :, :, 0]
Y = m[:, :, :, 1]

point_cloud = torch.tensor([10, 490, 490, 10, 250]).view([1,1,-1]), torch.tensor([10, 10, 490, 490, 250]).view([1,1,-1])

n_pointcloud = point_cloud[0]/(.5*img.size(-1))-1, point_cloud[1]/(.5*img.size(-2))-1

import tormentor



