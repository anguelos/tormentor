# Tormentor : pyTORch augMENTOR

Image data augmentation for pytorch

![Example](example.png)

### Instalation

bash```

```

### Design Principles

* Simplify the definition of augmentations
* Every instance of every augmentation class is deterministic.
* Inputs and Outputs are pytorch tensors and pytorch is prefered for all computation.
* All data are by default 4D: [batch x channel x width x height].
* Single sample augmentation: batch-size must always be 1.
* Threadsafety: Every augmentation instance must be threadsafe.
* Input/Output is restricted to one or more channels of 2D images.
* Augmentations either preserve channels or the preserve pixels (space).
* The augmentation class has also its factory as a classmethod
* Restrict dependencies on torch and kornia (at least for the core packages).

### Factory Dictates Constructor

In order to minimize the code needed to define an augmentation.
The factory defines the random distributions from wich augmentation sample.
The inherited constructor handles random seeds.
The method forward_sample_img samples from the random distributions aug_parameters and employs them.


### Internal Conventions

* Pointclouds are represented in image coordinates Sampling fields in normalised -1,1 coordinates
* By default we write code for batch processing
* Determinism is strictly handled by BaseAugmentation and all augment_*** methods.
* An augmentation must reside in a single device
* All randomness must be coming from pytorch
* Spatial augmentation samplingfields are normalised to -1, 1 so their effect magnitude is proporsional to image size (They are top down). 