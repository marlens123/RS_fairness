# README

## Description

This codebase investigates the influence of the pre-training dataset on fairness in remote sensing land cover segmentation. The pre-training datasets used are: ImageNet-1K, Satlas Sentinel-2, and Satlas Aerial (NAIP). We evaluate performance across rural and urban subgroups of the LoveDA dataset and compute mean maximum disrepancy (MMD) between the pre-training and evaluation datasets.

The project was conducted as final class project of the McGill course COMP-598-001 in the winter semester 2025.


## Credits
This codebase builds upon and integrates components from the following projects:

- [Satlas](https://github.com/allenai/satlaspretrain_models)

- [LoveDA](https://github.com/Junjue-Wang/LoveDA?tab=readme-ov-file)

```
    @inproceedings{NEURIPS DATASETS AND BENCHMARKS2021_4e732ced,
         author = {Wang, Junjue and Zheng, Zhuo and Ma, Ailong and Lu, Xiaoyan and Zhong, Yanfei},
         booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks},
         editor = {J. Vanschoren and S. Yeung},
         pages = {},
         publisher = {Curran Associates, Inc.},
         title = {LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation},
         url = {https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/4e732ced3463d06de0ca9a15b6153677-Paper-round2.pdf},
         volume = {1},
         year = {2021}
    }
    @dataset{junjue_wang_2021_5706578,
        author={Junjue Wang and Zhuo Zheng and Ailong Ma and Xiaoyan Lu and Yanfei Zhong},
        title={Love{DA}: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation},
        month=oct,
        year=2021,
        publisher={Zenodo},
        doi={10.5281/zenodo.5706578},
        url={https://doi.org/10.5281/zenodo.5706578}
    }
```

The project template is adapted from work by [Francis Pelletier / Rolnick Lab](https://github.com/RolnickLab).
The evaluation framework draws on methods developed by [Zhang et al. (2022)](https://openaccess.thecvf.com/content/CVPR2022W/FaDE-TCV/papers/Zhang_Segmenting_Across_Places_The_Need_for_Fair_Transfer_Learning_With_CVPRW_2022_paper.pdf).

We sincerely thank all original authors for making their work publicly available.
We also gratefully acknowledge the valuable feedback and support from our professor and TAs throughout the project.


## Initialization

## Note

This repository contains a submodule (satlaspretrain-models). To properly initiate the submodule run the following command:

´git submodule update --init --recursive´

## Python Version

This project uses Python version 3.10 and up.

## Build Tool

This project uses `poetry` as a build tool. Using a build tool has the advantage of 
streamlining script use as well as fix path issues related to imports.

## Quick setup

This is a short step by step, no nonsense way of setting things up and start working 
right away. 

For more in depth information, read the other sections below, starting at the 
[Detailed documentation section](#detailed-documentation).

**Reminder:** When working on the clusters, you will always need to load the 
appropriate module before you can activate your environment
* For python virtual environments: `module load python/<PYTHON_VERSION>`
* For conda environments : `module load miniconda/3`

### Install poetry

Skip this step if `poetry` is already installed. 

Installing `poetry` with `pipx` will make it available to all your other projects, so
you only need to do this once per system (i.e. on your computer, on the MILA cluster, etc.)

See [Installing Poetry as a Standalone section](docs/poetry_installation.md#installing-poetry-as-a-standalone-tool) 
 if working on a compute cluster.

1. Install pipx `pip install pipx`
2. Install poetry with pipx: `pipx install poetry`
   1. If installing poetry on DRAC, consider installing with `pipx install 'poetry<2.0.0'`. 
      See [Poetry version concerns on the clusters](docs/poetry_installation.md#version-concerns-on-drac)

### Create project's virtual environment

1. Read the documentation on the specific cluster if required:
   * [How to create a virtual environment for the Mila cluster](docs/environment_creation_mila.md)
   * [How to create an environment for the DRAC cluster](docs/environment_creation_drac.md) 
2. Create environment : `virtualenv <PATH_TO_ENV>`
   * Or, using venv : `python3 -m venv <PATH_TO_ENV>`
3. Activate environment : `source <PATH_TO_ENV>/bin/activate`

Alternatively, if you want or need to use `conda`:

1. Read the documentation about [conda environment creation](docs/conda_environment_creation.md)
2. Create the environment : `conda env create python=<PYTHON_VERSION_NUMBER> -n <NAME_OF_ENVIRONMENT>`
3. Activate environment : `conda activate <NAME_OF_ENVIRONMENT>`

### Install

1. Install your package : `poetry install`
2. Initialize pre-commit : `pre-commit install`

### Development

1. [Add required dependencies](./CONTRIBUTING.md#adding-dependencies)
2. Create some new modules in the [src](src/) folder!

## Detailed documentation

### Environment Management

This section and those following go into more details for the different setup steps.

Your project will need a virtual environment for your dependencies.

* [How to create a virtual environment for the Mila cluster](docs/environment_creation_mila.md)
* [How to create an environment for the DRAC cluster](docs/environment_creation_drac.md)
* [How to create a Conda environment](docs/conda_environment_creation.md)
* [Migrating to DRAC from another environment](docs/migrating_to_drac.md)

There are different ways of managing your python version in these environments. On the 
clusters, you have access to different python modules, and through `conda` you have access 
to practically all the python versions that exist. 

However, on your own system, if you do not wish to use `conda`, you will have to either 
manually install different versions of python manually for them to be usable by `poetry` 
or use a tool like [pyenv](https://github.com/pyenv/pyenv).

Do note that `conda` is not available on the DRAC cluster, and there are some extra steps
to use `conda` on the Mila cluster compared to a workstation.

Once you know in which environment you will be working, we can proceed to install `poetry`.

`poetry` can be installed a number of ways, depending on the environment choice, and if
you are working on your local machine vs a remote cluster. See the following 
documentation to help you determine what is best for you.

* [How to install poetry](docs/poetry_installation.md)
  * Do not skip the [Cluster recommendations with Poetry](poetry_installation.md#considerations-when-using-poetry-in-a-compute-cluster-environment)
    if developing directly on a compute cluster.

### Installation

Once the virtual environment is built and `poetry` is installed, follow these steps to 
install the package's dependencies:

1. Make sure your virtual environment is active
2. Install the package and its dependencies using the following command:
    * `poetry install`
    * Alternatively, you can also install using `pip install -e .`, which will install 
      your package, [configured scripts](https://python-poetry.org/docs/pyproject#scripts) 
      and dependencies, but without creating a `poetry.lock` file.

### Run Experiments

To run the transfer learning experiments, adjust the respective shell scripts to your compute environment and excute them.

For example, run ```run_aerial_eval.sh``` to evaluate all LoveDA with an aerial-US-pre-trained model, or ```run_mmd.sh``` to compute mean maximum disrepancy between the aerial-US pre-training set and the LoveDA subsets.