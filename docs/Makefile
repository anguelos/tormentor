# Minimal makefile for Sphinx documentation
#
# Based on kornia's documentation: https://github.com/kornia/kornia/blob/master/docs/Makefile

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = tormentor
SOURCEDIR     = source
BUILDDIR      = build

EXAMPLE_IMAGES = ./source/_static/example_images/AugmentationCascade.png ./source/_static/example_images/AugmentationChoice.png ./source/_static/example_images/Brightness.png ./source/_static/example_images/ColorJitter.png ./source/_static/example_images/Contrast.png ./source/_static/example_images/ElasticTransform.png ./source/_static/example_images/RemoveRectangle.png ./source/_static/example_images/Flip.png ./source/_static/example_images/Hue.png ./source/_static/example_images/Invert.png ./source/_static/example_images/PadTo.png ./source/_static/example_images/Perspective.png ./source/_static/example_images/PlasmaBrightness.png ./source/_static/example_images/PlasmaContrast.png ./source/_static/example_images/PlasmaRgbBrightness.png ./source/_static/example_images/PlasmaShadow.png ./source/_static/example_images/Rotate.png ./source/_static/example_images/Saturation.png ./source/_static/example_images/Scale.png ./source/_static/example_images/ScaleTranslate.png ./source/_static/example_images/Shred.png ./source/_static/example_images/Translate.png ./source/_static/example_images/Wrap.png ./source/_static/example_images/Zoom.png


# Put it first so that "make" without argument is like "make help".
help:
	$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile all_images
	echo $@
	$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

all_images: $(EXAMPLE_IMAGES)

# TODO(anguelos) add ./augmentation_example_list.json ./render_augmentation_examples.py as dependencies to $EXAMPLE_IMAGES
$(EXAMPLE_IMAGES):
	python3 render_augmentation_examples.py

.PHONY: help Makefile all_images

clean:
	@echo "Removing everything under 'build'.."
	rm -rf $(BUILDDIR)/html/ $(BUILDDIR)/doctrees
	rm -rf $(EXAMPLE_IMAGES)
