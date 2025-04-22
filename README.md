# README

## Description

This codebase investigates the influence of the pre-training dataset on fairness in remote sensing land cover segmentation. The pre-training datasets used are: ImageNet-1K, Satlas Sentinel-2, and Satlas Aerial (NAIP). We evaluate performance across rural and urban subgroups of the LoveDA dataset and compute mean maximum disrepancy (MMD) between the pre-training and evaluation datasets.

The project was conducted as final class project of the McGill course COMP-598-001 in the winter semester 2025.


## Credits
This codebase inherits code from satlas, licenced under Apache 2.0, and loveda, licensed under ....

The template of this project is based on RolnickLab/ Francis Pelletier, licenced under MIT.

The project builds on work by Zhang et al.

We thank all original authors for their contribution.

We also like to thank the prof and TA's for their helpful comments.


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

### Development

If you want to contribute to this repository, the development dependencies will also need to added.

1. Install `pre-commit` and other dev dependencies using the following command:
   * `poetry install --with dev`
     * `pre-commit` is used for code quality and code analysis
   * Configure `pre-commit` by running the following command: `pre-commit install`
   * To use manually on the project:
     * `nox`
     * `pre-commit run --all-files`
2. Optional Checks and Fixes with [Nox](https://nox.thea.codes/en/stable/)
   1. Pylint
      * While not enforced by the pre-commit tool, running Pylint on your code can help
        with code quality, readability and even catch errors or bad coding practices.
      * To run this tool : `nox -s pylint`
      * For more information, see the [Pylint library](https://pylint.readthedocs.io/en/stable/)
   2. Cyclomatic Complexity check (McCabe)
      * While not enforced by the pre-commit tool, running a complexity check on your code can help
        with code quality, readability and even catch errors or bad coding practices.
      * To run this tool : `nox -s complexity`
      * For more information, see [McCabe Checker](https://github.com/PyCQA/mccabe)
   3. Other `nox` options on the code base (use `nox -s <option>`)
      * `check` Runs all checks on the code base without modifying the code
      * `fix` : Runs the black, isort, docformatter and flynt tools on the code base
      * `flake8` : Runs the `flake8` linter
      * `black` : Runs the code formatter
      * `isort` : Runs the import sorter
      * `flynt` : Runs the `f-string formatter
      * `docformatter` : Runs the docstring formatter 
      * `test` : Runs tests found in the `tests/` folder with `pytest`
3. Python library dependencies
   * To keep things simple, it is recommended to store all new dependencies as main 
     dependencies, unless you are already familiar with dependency management. 
4. Read and follow the [Contributing guidelines](CONTRIBUTING.md)