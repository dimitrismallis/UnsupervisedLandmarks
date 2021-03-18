# Unsupervised Learning of Object Landmarks via Self-Training Correspondence (NeurIPS2020)
### Dimitrios Mallis, Enrique Sanchez, Matt Bell, Georgios Tzimiropoulos

This repository contains the training and evaluation code for our NeurIPS 2020 paper ["Unsupervised Learning of Object Landmarks via Self-Training Correspondence"](https://papers.nips.cc/paper/2020/file/32508f53f24c46f685870a075eaaa29c-Paper.pdf). The sofware learns a deep landmark detector, directly from raw images of a specific object category, without requiring any manual annotations.


![alt text](images/repo1.png "Method Description")


## Data Preparation


### CelebA

CelebA can be found [here](http://www.robots.ox.ac.uk/~vgg/research/unsupervised_landmarks/resources/celeba.zip). Download the .zip file inside an empty directory and unzip. We provide precomputed bounding boxes and 68-point annotations (for evaluation only) in _data/CelebA_.

### LS3D
We use [300W-LP](https://drive.google.com/file/d/0B7OEHD3T4eCkVGs0TkhUWFN6N1k/view?usp=sharing) database for training and [LS3D-balanced](https://www.adrianbulat.com/downloads/FaceAlignment/LS3D-W-balanced-20-03-2017.zip) for evaluation. Download the files in 2 seperate empty directories and unzip. We provide precomputed bounding boxes for 300W-LP in _data/LS3D_.


### Human3.6
Human3.6, database is availiable [here](http://vision.imar.ro/). From the availiable downloads we only requre video data, bounding boxes _(MySegmentsMat-> ground\_truth\_bb)_ and 2D keypoints for evaluation _(MyPoseFeatures-> D2\_Positions)_ . For easier download you can use an automated tool like [this](https://github.com/kotaro-inoue/human3.6m_downloader).

We provide a python script to preprocess the video data. Before executing the script ensure that download database follows the following path structure:

```
     training/
	 └── subjects/
		  └── S1/
		      └── Video/
		      |	    └── *.mp4
		      │
		      ├── MyPoseFeatures/
		      │		└── D2_Positions/
		      │		      └── *.cfd
		      └── MySegmentsMat/
			        └── ground_truth_bb/
			   	          └── *.mat  
```


To create the database please run:

```
python PrePreprocessHuman.py --path_to_extract_dataset <pathToHuman3.6_database> --path_to_Human <path_to_Human>
```

_\< path\_to\_Human \>_ is the directory where Human3.6 is downloaded. Frames, bounding boxes and 2D point annotations (for evaluation only) will be extracted in _\< pathToHuman3.6\_database \>_. 


## Installation

You require a reasonable CUDA capable GPU. This project was developed using Linux. 

Create a new conda environment and activate it:

```
conda create -n UnsuperLandmEnv python=3.8
conda activate UnsuperLandmEnv
```

Install [pythorch](https://pytorch.org/) and the [faiss library]((https://github.com/facebookresearch/faiss) ):

```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install -c pytorch faiss-gpu cudatoolkit=10.2
```

Install other external dependencies using pip.

```
pip install -r requirements.txt 
```



Our method is bootstraped by Superpoint. Download weights for a pretrained Superpoint model from [here](https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/superpoint_v1.pth).

Before code execution you have to update `paths/main.yaml` so it includes all the required paths. Edit the following entries in `paths/main.yaml`.:

```
CelebA_datapath: <pathToCelebA_database>/celeb/Img/img_align_celeba_hq/
300WLP_datapath: <pathTo300W_LP_database>/300W_LP/
LS3Dbalanced_datapath: <pathToLS3D-balanced_database>/LS3D-balanced/
Human_datapath: <pathToHuman3.6_database>/
path_to_superpoint_checkpoint: <pathToSuperPointCheckPoint>/superpoint_v1.pth
```




## Training

To execute the first step of our method please run:

```
python Train_Firststep.py --dataset_name <dataset_name> --experiment_name <experiment_name>
```

Similarly, to execute the second step please run:

```
python Train_Secondstep.py --dataset_name <dataset_name> --experiment_name <experiment_name>
```

where _\< dataset\_name \>_ is in ``["CelebA","LS3D", "Human3.6"]`` and _\< experiment\_name \>_ is a custom name you choose for each experiment. Please use the **same experiment name for both the first and second step**. The software will automatically initiate the second step with the groundtruth descovered in step one.

## Testing
To evaluate the trained model simply execute:

```
python Test.py --dataset_name <dataset_name> --experiment_name <experiment_name>
```

The script will calculate cumulative forward and backward error curves. Will be stored in _Logs/\<experiment\_name\>/Logs/_ .


## Visualisations
We provide 3 different visualisations.

### Keypoints (Step 1):
To inspect keypoint 2D locations learned from the detector head without without correspondance run:

```
python Visualise.py --visualisation Step1_Keypoints --dataset_name <dataset_name> --experiment_name <experiment_name> 
```

![alt text](images/repo_keypoints.png "Example of detected keypoints.")

### Clusters (Step 1):
To inspect examples of keypoints assigned to the same cluster run:

```
python Visualise.py --visualisation Step1_Clusters --dataset_name <dataset_name> --experiment_name <experiment_name> 
```

![alt text](images/repo_cluster.png "Example of keypoints assigned to the same cluster.")

This will create a .jpg file per cluster.

### Visual Results (Step 2):
For visual results run:

```
python Visualise.py --visualisation Step2 --dataset_name <dataset_name> --experiment_name <experiment_name> 
```

![alt text](images/repo_results.png "Visual results.")

The software will automatically load checkpoints and pseudogroundtruth files for the assosiated `<experiment_name> `.



## Pretrained Models

We provide also pretrained models. Can be used to execute the testing script and produce visual results.

| Dataset       |Experiment_name |Model        
| ------------- |:----------| --------------- |
| **CelebA**   | _CelebA\_pretrained_ |   [link](https://drive.google.com/file/d/1pPSUIhImP5G__9k9aGwZzbrwOhxKtyfF/view?usp=sharing) |
| **LS3D**      | _LS3D\_pretrained_ |   [link](https://drive.google.com/file/d/14iF5ISS00Z47up7KFyW85R9_NjtcRInV/view?usp=sharing) |
| **Human3.6**   |  _Human\_pretrained_ | [link](https://drive.google.com/file/d/1q5fEYNgg4O-Ka4sNL4PDm3IJ4czN69Tl/view?usp=sharing) |

Simply uncompress the .zip files inside `Logs/`.

Pretrained weights can be used for calculating forward and backward error curves as well as running visualisation code for **visual results (step 2)**.

## Citation
If you found this code useful please consider citing:

```
@inproceedings{unsupervLandm2020,
title={Unsupervised Learning of Object Landmarks via Self-Training Correspondence},
author={Mallis, Dimitrios and Sanchez, Enrique and Bell, Matt and Tzimiropoulos, Georgios},
booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
year={2020}
}
```
