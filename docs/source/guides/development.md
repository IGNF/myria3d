# Developer's guide

## Code versionning

Package version follows semantic versionning conventions and is defined in `package_metadata.yaml`. 

Releases are created when new high-level functionnality are implemented (e.g. a new step in the production process), with a documentation role. A `prod-release-tag` is created that tracks an _arbitrary_ commit, and serves as a mean to make a few models, model card, and config accessible via its associated [release](https://github.com/IGNF/myria3d/releases/tag/prod-release-tag).

## Tests

Tests can be run in an activated environment with.

```bash
conda activate myria3d
python -m pytest -rA -v
```

## Continuous Integration (CI)

New features are developped in ad-hoc branches (e.g. `2023MMDD-Feature-Name`).

CI tests are run for push and pull request on the `main` branche. The workflow builds a docker image, runs linting, and tests the code.

## Continuous Delivery (CD)

In case of push / accepted merge to the `main` branch, and if the CI workflow is successful (i.e. docker build is complete, tests pass, and code is PEP8 compliant), a docker image is pushed to an in-house Nexus image repository.

Additionnaly, images may be built for feature branches, for further testings / staging. Details are in workflow `cicd.yaml`.

See [../tutorials/use.md] for how to leverage such image to run the app.

Additionnaly, pushes on the `main` branch build this library documentation, which is hosted on Github pages.
