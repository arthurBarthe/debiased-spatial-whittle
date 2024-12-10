The package can be installed via one of the following methods.

## pip

    pip install debiased-spatial-whittle

If you want to install the optional Cupy and Pytorch dependencies for GPU computations,
you need to add the optional gpu dependency:

    pip install debiased-spatial-whittle[gpu]

Otherwise, you should be able to directly install this repository:

    pip install git@github.com:arthurBarthe/debiased-spatial-whittle

## Poetry
Via the use of Poetry (https://python-poetry.org/) by running

    poetry add debiased-spatial-whittle

Or, to add optional gpu dependencies,

    poetry add debiased-spatial-whittle -E gpu

You can also use the latest github version by adding the following dependency in your pyproject.toml:

    debiased-spatial-whittle = {git = "git@github.com:arthurBarthe/debiased-spatial-whittle", branch="master"}

## Development

Install for development - in this case you need to clone the github repository and (assuming you already have poetry installed) run
```
poetry install
```
which will install an environment based on the dependencies listed in the lock
file.

To install with gpu dependencies,

    poetry install -E gpu
