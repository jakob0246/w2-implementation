# EigenCam for YOLO object detection

This is a demo of a GradCam output on a YOLOv8s model. This particular network was trained to detect ice blocks in
the Mars North Polar region. If you wish to run the <em>camtest.ipynb</em> yourself I first suggest running:

```
pip install ultralytics
```

This will install additional yolo dependencies on top of those required for EigenCam (needed to load the model using the YOLO wrapper).

### Results

OG Images             |  EigenCam
:-------------------------:|:-------------------------:
![alt text](test_images/46920_12260.png "...") | ![alt text](results/46920_12260_box_cam.png "...")
![alt text](test_images/56680_11880.png "...") | ![alt text](results/56680_11880_box_cam.png "...")
![alt text](test_images/81800_19000.png "...") | ![alt text](results/81800_19000_box_cam.png "...")
![alt text](test_images/86760_20460.png "...") | ![alt text](results/86760_20460_box_cam.png "...")
![alt text](test_images/99400_20780.png "...") | ![alt text](results/99400_20780_box_cam.png "...")
