# Backdoor-Trigger-Detection
Benchmark and code for Backdoor Trigger Detection (BTD)

### Dataset
Please download our benchmar in our [Anonymous Github](https://anonymous.4open.science/r/Backdoor-Trigger-Detection-52F0). Please put the unzipped files in `record/`.


### Environment
The requirements of the environment are provided in `environment.yaml`. You can create the environment by
```
conda env create -f environment.yaml
```


### Method
Our algorithm for detecting triggers in images is provided in `TriDet/image_detector.py`. Our algorithm for detecting triggers in texts is provided in `TriDet/text_detector.py`. 

### Run Experiments
You can run image experiments by 
```
python run_image.py --record_dir record/BTD-CIFAR-10/BadNets_0/
```
where `record_dir` is path of the test set.

You can run text experiments by 
```
python run_text.py --record_dir record/BTD-SST-2/BadNets_0/
```
where `record_dir` is path of the test set.
