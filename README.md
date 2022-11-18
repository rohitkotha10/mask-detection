# mask-detection

[Kaggle Dataset for Mask images](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

## Build instructions
- use `pip install -r requirements.txt`
- you need to get the images for dataset from the above link and copy it to `dataset` folder 
- train the model by running `my_trainer.py`
- test on image using `python mask_image.py --image <image-path>`
- test on webcam using `python mask_video.py`