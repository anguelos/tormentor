Text Segmentation

Pixel classification into text not text 

Scene-text segmentation:
For the dataset download the following
* [Trainset inputs](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=2&f=aHR0cHM6Ly9ycmMuY3ZjLnVhYi5lcy9kb3dubG9hZHMvQ2hhbGxlbmdlMl9UcmFpbmluZ19UYXNrMTJfSW1hZ2VzLnppcA==)
* [Trainset groundtruth](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=2&f=aHR0cHM6Ly9ycmMuY3ZjLnVhYi5lcy9kb3dubG9hZHMvQ2hhbGxlbmdlMl9UcmFpbmluZ19UYXNrMl9HVC56aXA=)
* [Testset inputs](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=2&f=aHR0cHM6Ly9ycmMuY3ZjLnVhYi5lcy9kb3dubG9hZHMvQ2hhbGxlbmdlMl9UZXN0X1Rhc2sxMl9JbWFnZXMuemlw)
* [Testset groundtruth](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=2&f=aHR0cHM6Ly9ycmMuY3ZjLnVhYi5lcy9kb3dubG9hZHMvQ2hhbGxlbmdlMl9UZXN0X1Rhc2syX0dULnppcA==)

Registration is required



```bash
PYTHONPATH="./" nice -n 20  python3 ./examples/text_segmentation/inference.py -inputs ~/data/rr/focused_segmentation/tmp/img_*.jpg -flip=False -device=cpu
```