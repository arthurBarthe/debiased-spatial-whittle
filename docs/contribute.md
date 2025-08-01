Contributions are most welcome, thank you for your interest! A good starting point is to take a look at
the list of [issues](https://github.com/arthurBarthe/debiased-spatial-whittle/issues).

## Development

The general steps to follow to make a contribution are:
1. **Fork the Repository**: You create your own copy of the repository by forking it. This allows you to make changes without affecting the original repository.

2. **Clone the Fork**: You then clone your fork to your local machine to work on the changes.

3. **Create a Branch**: This keeps your changes organized and makes it easier to manage multiple contributions.

4. **Make Changes**: You make your changes in your local branch.

5. **Commit Changes**: You commit your changes with descriptive commit messages.

6. **Push Changes**: You push your changes to your fork on the remote repository.

7. **Create a Pull Request**: Finally, you create a pull request from your fork to the original repository. This allows the maintainers to review your changes and merge them if they are acceptable.

Note that once you have cloned the repository, you will need to also install [poetry](https://python-poetry.org/),
which we use for packaging and dependency managment.

Having done so, you can create a virtual environment and install the package by running the following command from
the directory corresponding to the package:

   ```bash
      poetry install
   ```

If you run into some issue regarding the Python version, you can run
   ```bash
      poetry env use <path_to_python>
   ```
where <path_to_python> is the path to a Python version compatible with the requirements in pyproject.toml.

### Unit tests
Unit tests are run with pytest. On Pull-requests, the unit tests will be
run. They can be run locally via the command

```bash
   pytest
```
from the root of the directory, with your environment activated.

### Pre-commits
We run ruff-format as a pre-commit hook. Specifically, this is set up via pre-commit in
the `.pre-commit-config.yaml` file.

When you first try to commit, ruff-format might
make formatting changes to the code. You need to check those and then you can git-add again
those files and try to commit with those changes.

### Building the documentation
The documentation is built using mkdocs. You can build and serve the documentation
locally via the following command:

```bash
   mkdocs serve
```

### Versioning
Currently, versioning is handled manually using poetry, e.g.

   ```bash
      poetry version patch
   ```
or
   ```bash
      poetry version minor
   ```

When creating a release in Github, the version tag should be set to match
the version in th pyproject.toml. Creating a release in Github will trigger
a Github workflow that will publish to Pypi (see Pypi section).

## Bug reports

If you encounter a bug, we would be thankful if you could report it by raising an issue.
Before you submit an issue, please ensure that you are using the latest available version.

## Pull request

To propose changes to the main branch, you can submit a pull request. Note that those
changes should not fall outside the main scope of the package.

Before a PR can be accepted, the following criteria are required:
1. All unit tests should pass
2. New functionality should come with new unit tests
3. Ruff checks and format should pass
4. New functionality should come with docstrings and a corresponding section in the mkdocs
documentation.
5. If a PR addresses a specific issue, the latter should be referenced in the PR's description

Depending on the type of contribution (patch, minor, major), the version will be
updated accordingly.
