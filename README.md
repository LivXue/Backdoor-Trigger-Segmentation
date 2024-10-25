# Backdoor Trigger Segmentation
Benchmark and code for Backdoor Trigger Segmentation (BTS)

### Benchmark
Please download our benchmark (83GB) from [Google Drive](https://drive.google.com/drive/folders/1u09aO7S81Us50_U_RAyKMTCe5LuIA5Ut?usp=sharing) or [Badidu Netdisk](https://pan.baidu.com/s/1TF2EU12pxjt1-KBBYtReUQ?pwd=v1xu) (password: v1xu). Please put the unzipped files in `record/`.


### Environment
The requirements of the environment are provided in `environment.yaml`. You can create the environment by
```
conda env create -f environment.yaml
```


### Method
Our algorithm for segmenting triggers in images is provided in `TriLoc/image_locator.py`. Our algorithm for segmenting triggers in texts is provided in `TriLoc/text_locator.py`. 


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