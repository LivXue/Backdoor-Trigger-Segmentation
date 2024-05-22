# Backdoor-Trigger-Detection
Benchmark and code for Backdoor Trigger Detection (BTD)

### Dataset
We release the BTD dataset at [Baidu Netdisk](https://xxxxx). Please put the unzipped files in `record/`.


### Environment
The requirements of the environment are provided in `environment.yaml`. You can create the environment by
```
conda env create -f environment.yaml
```


### Method
Our algorithm for detecting triggers in images is provided in `TriDet/image_detector.py`. Our algorithm for detecting triggers in texts is provided in `TriDet/text_detector.py`. 

### Run Experiments
To 