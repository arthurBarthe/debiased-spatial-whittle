The package can be installed via one of the following methods. Note that in
all cases, since the repository is currently private, git needs to be configured
on your machine with an SSH key linking your machine to your github account.

## Poetry
Via the use of Poetry (https://python-poetry.org/), by adding
the following line to the dependencies listed in the pyproject.toml
of your project:


    debiased-spatial-whittle = {git = "git@github.com:arthurBarthe/dbw_private.git", branch="master"}


## pip
Otherwise you should be able to directly install via pip:


    pip install git+https://github.com/arthurBarthe/dbw_private.git

## Developement

Install for developement - in this case you need to clone the github repository and (assuming you already have poetry installed) run
```
poetry install
```
which will install an environment based on the dependencies listed in the lock
file.