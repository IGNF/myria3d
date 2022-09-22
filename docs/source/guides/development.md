# Developer's guide

## Code versionning

Package version follows semantic versionning conventions and is defined in `setup.py`. 

Releases are generated when new high-level functionnality are implemented (e.g. a new step in the production process), with a documentation role. Production-ready code is fast-forwarded in the `prod` branch when needed to match the `main` branch. When updating the `prod` branch, one should move the tag `prod-release-tag` alongside to the [related release](https://github.com/IGNF/myria3d/releases/tag/prod-release-tag).

## Tests

Tests can be run in an activated environment with.

```bash
conda activate myria3d
python -m pytest -rA -v
```

## Continuous Integration (CI)

New features are developped in ad-hoc branches (e.g. `dev-Vx.y.z-Feature-Name`).

CI tests are run for pull request to merge on either the `main` branches, and on pushes to `main`, and `prod` branches. The CI workflow builds a docker image, runs linting, and tests the code.

## Continuous Delivery (CD)

When the event is a push and not a merge request, this means that there was either a direct push to `main`|`prod` or that a merge request was accepted. In this case, if the CI workflow passes (i.e. tests pass and code is PEP8 compliant), the created docker image is tagged with the branch name, resulting in e.g. a `myria3d:prod` image that is up to date with the branch content. 
See [../tutorials/use.md] for how to leverage such image to run the app.

Additionnaly, pushes on the `main` branch build this library documentation, which is hosted on Github pages.


