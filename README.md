# Drone-Detection
Detect drones from input image, video or real-time feed.

## Requirements

- Refer to requirement.txt for environment specifications.
- Download the pre-trained YOLOv3 weights from [here.](https://pjreddie.com/media/files/yolov3.weights)
- Download an image of a dog to test object detection.

  > ```shell 
  > $ python yolo3_one_file_to_detect_them_all.py -w yolo3.weights -i dog.jpg 
  > ```
- Download pretrained weights for backend from [here.](https://1drv.ms/u/s!ApLdDEW3ut5fgQXa7GzSlG-mdza6) This weight must be put in the root folder of the repository. 

## Dataset
YOLOv3 training requires images with .xml files in PASCAL-VOC format.

Click [here] to Download Drone Dataset with .xml files in PASCAL-VOC format.

Alternatively, if you want to create your own dataset, follow these steps:
   1. Collect images from Kaggle Dataset or Google Images.
   2. Download LabelImg(a graphical image annotation tool) from [this GitHub Repo.](https://github.com/tzutalin/labelImg)
   3. Setup LabelImg and draw a box around the object of interest in each image using the tool to generate XML files.
   4. Place all your dataset images in the **images** folder and the xml files in the **annots** folder.

## Training

### 1. Edit config.json

- Specify path of the **images** and **annots** folder in the `"train_image_folder"` and `"train_annot_folder"` fields.
- The `"labels"` setting lists the labels to be trained on. Only images, which has labels being listed, are fed to the network.

```sh
{
    "model" : {
        "min_input_size":       288,
        "max_input_size":       448,
        "anchors":              [17,18, 28,24, 36,34, 42,44, 56,51, 72,66, 90,95, 92,154, 139,281],
        "labels":               ["drone"]
    },

    "train": {
        "train_image_folder":   "F:/Drone/Drone_mira_dataset/images/", 
        "train_annot_folder":   "F:/Drone/Drone_mira_dataset/annots/",
        "cache_name":           "drone_train.pkl",

        "train_times":          8,     # the no. of times to cycle through the training set
        "pretrained_weights":   "",    # specify path of pretrained weights,but it's fine to start from scratch       
        "batch_size":           2,     # the no. of images to read in each batch
        "learning_rate":        1e-4,  # the base learning rate of the default Adam rate scheduler
        "nb_epochs":            50,    # no. of epoches
        "warmup_epochs":        3,       
        "ignore_thresh":        0.5,
        "gpus":                 "0,1",

        "grid_scales":          [1,1,1],
        "obj_scale":            5,
        "noobj_scale":          1,
        "xywh_scale":           1,
        "class_scale":          1,

        "tensorboard_dir":      "logs",
        "saved_weights_name":   "drone.h5", # name of model file to which our trained model is saved
        "debug":                true    # turn on/off the line to print current confidence,position,size,class losses,recall
    },

    "valid": {
        "valid_image_folder":   "C:/drone/valid_image_folder/",
        "valid_annot_folder":   "C:/drone/valid_annot_folder/",
        "cache_name":           "drone_valid.pkl",

        "valid_times":          1
    }
}
```

### 2. Generate anchors for your dataset
   > ```shell 
   > $ python gen_anchors.py -c config.json
   > ```
Copy the generated anchors printed on the terminal to the anchors setting in config.json.

### 3. Start the training process
   > ```shell 
   > $ python train.py -c config.json
   > ```
By the end of this process, the code will write the weights of the best model to file drone.h5 (or whatever name specified in the setting "saved_weights_name" in the config.json file). The training process stops when the loss on the validation set is not improved in 3 consecutive epoches.

### 4. Perform detection using trained weights on image, set of images, video, or webcam
   > ```shell 
   > $ python predict.py -c config.json -i /path/to/image/or/video/or/cam
   > ```
- For an image use : `$ python predict.py -c config.json -i test.jpg`
- For a video  use : `$ python predict.py -c config.json -i test.mp4`
- For a real-time feed use : `$ python predict.py -c config.json -i webcam`

It carries out detection on the image and write the image with detected bounding boxes to the output folder.

## Evaluation
Compute the mAP performance of the model defined in saved_weights_name on the validation dataset defined in `"valid_image_folder"` and `"valid_annot_folder"`  
   > ```shell 
   > $ python evaluate.py -c config.json
   > ```

## OUTPUT

Demo:

![](https://github.com/harshiniKumar/Drone-Detection-using-YOLOv3/blob/master/Outputs/Drone-Detection-Demo.gif)
- Download the [sample output for drone detection in a video.](https://github.com/harshiniKumar/Drone-Detection-using-YOLOv3/blob/master/Outputs/Drone_Video_Detection.mp4?raw=true)
