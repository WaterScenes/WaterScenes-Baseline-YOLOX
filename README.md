## YOLOX-VR for WaterScenes Benchmark
---


## Data Preparation
1. Prepare the data in the format of 2007_train.txt and 2007_val.txt
2. Define the class_path and radar_file_root

```python
classes_path    = 'model_data/waterscenes_benchmark.txt'
radar_file_path = "your own path"
```
3. Train the model. All hyperparameters are in the train.py
```python
python train.py
```

4. Test the model. 
   1. Modify various file paths.
   2. run predict.py
```python
"model_path"        : r'model_data\yolov8_vr.pth',
"classes_path"      : r'model_data\waterscenes_benchmark.txt',
"radar_root": r"your radar map root",
```


## Acknowledgement
https://github.com/bubbliiiing/yolox-pytorch
