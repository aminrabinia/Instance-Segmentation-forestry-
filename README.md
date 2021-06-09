# Instance-Segmentation-forestry

This project aims to automate the measurement of woody debris in forest images using Instance Segmentation. The model used for this project is [PixelLib](https://github.com/ayoolaolafenwa/PixelLib) (an open source image processing library built on TensorFlow). The model is trained on labeled images (using [LabelMe](https://github.com/wkentaro/labelme)) and tested on unseen images to find any instance of fallen logs on the forest ground. The input images are a blend of four layers, from the same location during different time periods, using [OpenCV](https://opencv.org/). Images from different years are merged together with equal weights:

```
file_name = [ '2014', '2016', '2017','2018'] 

alpha = 0.5
beta = (1.0 - alpha)

img1 = cv2.imread(path + file_name[0] + '.jpg')
img2 = cv2.imread(path + file_name[2] + '.jpg')
out1 = cv2.addWeighted(img1, alpha, img2, beta, 0)

img3 = cv2.imread(path + file_name[1] + '.jpg')
img4 = cv2.imread(path + file_name[3] + '.jpg')
out2 = cv2.addWeighted(img3, alpha, img4, beta, 0)

out = cv2.addWeighted(out1, alpha, out2, beta, 0)
```

Below are four image samples from the same location during different time periods:

![image](https://user-images.githubusercontent.com/34719495/121390681-9bcaef00-c91b-11eb-88b8-060e473f9477.png)

After merging images the output looks like below. You can see how stationary objects (fallen logs which are our targets) stand out and the non-stationary objects (standing trees and shadows) look pale. Blended images like this are used during the training process. 

![image](https://user-images.githubusercontent.com/34719495/121392412-5c9d9d80-c91d-11eb-8450-55bfe7c0d0e5.png)



After manual annotation of such images with LabelMe, and splitting the data to train/eval, the model is trained for 20 epochs (backbone resnet 101 and pretrained weights mask_rcnn_coco). 
```
train_maskrcnn.modelConfig(network_backbone = "resnet101", num_classes= 1, batch_size = 2)
```

Data augmentation is also provided during the training. 
```
augmentation = imgaug.augmenters.Sometimes(0.9, [
			        imgaug.augmenters.Fliplr(0.5),
			        imgaug.augmenters.Flipud(0.5),
              imgaug.augmenters.Rot90([1,3]),
              imgaug.augmenters.Affine(rotate=[20,40,60,80]),
			        imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
			        ])
```

Below is an example of the blended image (left) and the automatically annotated logs (right). 

![image](https://user-images.githubusercontent.com/34719495/119857439-90ba9c80-bee1-11eb-9e0c-511287c15c6c.png)
