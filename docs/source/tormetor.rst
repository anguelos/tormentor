tormentor
=========

.. currentmodule:: tormentor

The functions in this sections perform various image filtering operations.


Sampling Fields
---------------
.. autofunction:: create_sampling_field
.. autofunction:: apply_sampling_field


Abstract Augmentation Types
---------------------------
.. autoclass:: DeterministicImageAugmentation
.. autoclass:: SpatialImageAugmentation
.. autoclass:: StaticImageAugmentation
.. autoclass:: ColorAugmentation
.. autoclass:: AugmentationCascade
.. autoclass:: AugmentationChoice


SpatialAugmentations
--------------------

.. autoclass:: Perspective
.. autoclass:: ExtendedRotate
.. autoclass:: Rotate
.. autoclass:: Zoom
.. autoclass:: Scale
.. autoclass:: Translate
.. autoclass:: ScaleTranslate
.. autoclass:: Flip
.. autoclass:: EraseRectangle
.. autoclass:: ElasticTransform


