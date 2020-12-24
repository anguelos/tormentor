import os
os.environ["CUDA_VISBLE_DEVICES"]=""
import torch
import fargv
from matplotlib import pyplot as plt
import numpy as np

from scipy.ndimage.filters import gaussian_filter1d

p={"models":set([]),
   "max_epoch": 200,
   "sigma":15,
}

p,_=fargv.fargv(p, return_named_tuple=True)
f,ax = plt.subplots(2,2)
for model in p.models:
   print("MODEL",model)
   pt_dict=torch.load(model,map_location="cpu")
   train_error = pt_dict["per_epoch_train_errors"]
   val_error = pt_dict["per_epoch_validation_errors"]
   for ax_n, (name, error) in enumerate([["Train Error", train_error], ["Patch Validation Error", val_error]]):
      Y=[]
      X=[]
      for x in sorted(error.keys()):
         X.append(x)
         Y.append(error[x][0])
      X=np.array(X)
      Y=np.array(Y)
      #print(X)
      #print(Y)
      ax[0][ax_n].plot(X[X<p.max_epoch],gaussian_filter1d((1-Y[X<p.max_epoch]),sigma=p.sigma),".-",label=model)
      ax[0][ax_n].set_ylabel(name)
      ax[0][ax_n].set_xlabel("Epoch")
      Y=[]
      X=[]
      for x in sorted(error.keys()):
         X.append(x)
         Y.append(error[x][3])
      X=np.array(X)
      Y=np.array(Y)
      #print(X)
      #print(Y)
      ax[1][ax_n].plot(X[X<p.max_epoch],(Y[X<p.max_epoch]),".-",label=model)

ax[0][0].legend()
ax[1][0].legend()
ax[0][1].legend()
ax[1][1].legend()
plt.show()

