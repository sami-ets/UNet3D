# <img src="/icons/artificial-intelligence.png" width="60" vertical-align="bottom"> UNet3D

## Welcome to UNet3D

![GitHub issues](https://img.shields.io/github/issues/sami-ets/UNet3D.svg)
![GitHub](https://img.shields.io/github/license/sami-ets/UNet3D.svg)
![GitHub contributors](https://img.shields.io/github/contributors/sami-ets/UNet3D.svg)


UNet3D is a project based on [SAMITorch library](https://github.com/sami-ets/SAMITorch/) of *Shape Analysis in Medical Imaging* laboratory of [École de technologie supérieure](https://www.etsmtl.ca/) using [PyTorch](https://github.com/pytorch) library
for establishing baselines for medical image segmentation tasks. The objective of this repository is to build a tested, standard UNet3D framework for quickly producing results and/or baseline metrics in deep learning reasearch applied to medical imaging. 

# Table Of Contents

-  [Authors](#authors)
-  [References](#references)
-  [Project architecture](#project-architecture)
    -  [Folder structure](#folder-structure)
 -  [Contributing](#contributing)
 -  [Branch naming](#branch-naming)
 -  [Commits syntax](#commits-syntax)
 -  [Acknowledgments](#acknowledgments)
 
 
## Authors

* Pierre-Luc Delisle - [pldelisle](https://github.com/pldelisle) 

## References
``
@article{RN10,
   author = {Çiçek, Özgün and Abdulkadir, Ahmed and Lienkamp, Soeren S. and Brox, Thomas and Ronneberger, Olaf},
   title = {3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation},
   journal = {eprint arXiv:1606.06650},
   pages = {arXiv:1606.06650},
   url = {https://ui.adsabs.harvard.edu/\#abs/2016arXiv160606650C},
   year = {2016},
   type = {Journal Article}
}
``


## Setup
> pip install -r [path/to/requirements.txt]  
> python3 <main_script>.py --config=UNet3D/config/config.yaml

If distributed : 
> sh run.sh

## Project architecture
### Folder structure

```
├── tests                   - Contain a basic model helper for testing purpose.
│   ├── models                  
│       └── model_helper_test.py   
|
├── icons                   - Contains project's artwork.
|
├── data                    - Contains Dockerfile needed to provide a functional Docker environment for your publication.
|   └── MRBrainS_2013
|       └── TrainingData
|           └── Source      - Contains source data.
|           └── Target      - Contains target data. 
|
├── UNet3D                  - Main project folder.  
|   └── config              - Configuration folder.
|   |   └── config.yaml
│   |
|   └──factories            - This folder contains anything relative to the creation of objects needed.
|   |   └── parsers.py      - Parse the configuration file elements. 
|   |
|   └── inputs              - Anything related to Inputs.
|   |   └── pipelines.py    - Preprocessing pipelne. Executed once.  
|   |
|   └── trainig             - Contains training logic. 
|   |   └── model_traininer.py      - Train one model, UNet3D here.       
|   |   └── trainer.py              - High level trainer object.
|   |
|   └── utils               - Various utility modules.
|       └── initializer.py  - Initialize the whole training process.
|       └── logger.py       - Logger object to log to Tensorboard.
|       └── utils.py        - Varioous utility methods/objects.
```

### Main components
This project is implemented with the high performance [APEX Library](https://github.com/NVIDIA/apex). APEX must be compiled prior running the training script.
Default optimization is 'O2'. See [APEX optimization levels](https://nvidia.github.io/apex/amp.html#opt-levels) for more details.

## Contributing
If you find a bug or have an idea for an improvement, please first have a look at our [contribution guideline](https://github.com/sami-ets/SAMITorch/blob/master/CONTRIBUTING.md). Then,
- [X] Create a branch by feature and/or bug fix
- [X] Get the code
- [X] Commit and push
- [X] Create a pull request

## Branch naming

| Instance        | Branch                                              | Description, Instructions, Notes                   |
|-----------------|-----------------------------------------------------|----------------------------------------------------|
| Stable          | stable                                              | Accepts merges from Development and Hotfixes       |
| Development     | dev/ [Short description] [Issue number]             | Accepts merges from Features / Issues and Hotfixes |
| Features/Issues | feature/ [Short feature description] [Issue number] | Always branch off HEAD or dev/                     |
| Hotfix          | fix/ [Short feature description] [Issue number]     | Always branch off Stable                           |

## Commits syntax

##### Adding code:
> \+ Added [Short Description] [Issue Number]

##### Deleting code:
> \- Deleted [Short Description] [Issue Number]

##### Modifying code:
> \* Changed [Short Description] [Issue Number]

##### Merging branches:
> Y Merged [Short Description]

## Acknowledgment
Thanks to [École de technologie supérieure](https://www.etsmtl.ca/), [Hervé Lombaert](https://profs.etsmtl.ca/hlombaert/) and [Christian Desrosiers](https://www.etsmtl.ca/Professeurs/cdesrosiers/Accueil) for providing us a lab and helping us in our research activities.

Icons made by <a href="http://www.flaticon.com/authors/freepik" title="Freepik">Freepik</a> from <a href="http://www.flaticon.com" title="Flaticon">www.flaticon.com</a> is licensed by <a href="http://creativecommons.org/licenses/by/3.0/" title="Creative Commons BY 3.0" target="_blank">CC 3.0 BY</a>
