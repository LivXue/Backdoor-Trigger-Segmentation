# Backdoor Trigger Detection
Benchmark and code for Backdoor Trigger Detection (BTD)

### Dataset
Please download our dataset from [Badidu Netdisk](https://pan.baidu.com/s/1TF2EU12pxjt1-KBBYtReUQ?pwd=v1xu) (password: v1xu) or [Google Drive](xxx) (Uploading). Please put the unzipped files in `record/`.


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


### Attacks
We implement image backdoor attacks by [BackdoorBench](https://github.com/SCLBD/BackdoorBench) and text backdoor atatcks by [OpenBackdoor](https://github.com/thunlp/OpenBackdoor). We include the attack code in `attack/` and `openbackdoor/attackers/`. However, all attack results are saved in our benchmark and these code is not used in BTD experiments. So you don't need to worry about their requirements.