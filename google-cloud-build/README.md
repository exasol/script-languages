# GCloud CI Setup

This setups a CI Builds with Google Cloud Build. It uses Triggers to automate different tasks, such as Build Images, Pushing Images to a Docker Registry, Run Tests, Export Container to Google Cloud Storage and Createing a Github Release. For this, it uses Google Cloud Build Triggers and Cloud Functions.

**Note: These scripts setup billable components, such as Google Cloud Build, Google Cloud KMS and Google Cloud Functions.**

## Requirements

- [GCloud SDK](https://cloud.google.com/sdk/docs/quickstarts)
- [jq](https://stedolan.github.io/jq/)
- [yq](https://pypi.org/project/yq/)
- [jinja2-cli with yaml support](https://pypi.org/project/jinja2-cli/)

## Setup

1. [Create a mirror of your Github Repository on Google Cloud Source Repositories](https://cloud.google.com/source-repositories/docs/mirroring-a-github-repository)
2. Setup triggers and functions
    1. gcloud auth login
    2. cp .env/env.template.yaml .env/env.yaml
    3. set config variables in .env/env.yaml
    4. ./setup.sh

## Usage

The following Triggers get created:

- `Build branch <flavor>`
  - Builds and tests the flavor for branches with the prefix `^(feature|bug|enhancement|refactoring|ci)/.*`
  - Intended for branches created by maintainers
  - Provides secrets to the build to push images to the Docker Registry
  - Uses cached images if possible for faster builds
- `Build pull request <flavor>`
  - Builds and tests the flavor for branches with the prefix `^pull_request/.*`
  - Intented for pull requests from external contributors
  - Does not provide any secrets to the build
  - Won't push images to the Docker Registry
  - Uses cached images if possible for faster builds
- `Build release <flavor>`
  - Builds ands tests the flavor for branches with the pattern `^develop$`
  - Intended for branches created by maintainers
  - Provides secrets to the build to push images to the Docker Registry
  - Rebuilds all images
- `Rebuild <flavor>`
  - By default disabled, but can be triggered manually for any branch
  - Rebuilds all images and run tests
  - Intended for branches created by maintainers
  - Provides secrets to the build to push images to the Docker Registry
- `Release`
  - Exports the container to Google Cloud Storage, Pushes the images to public build cache and create the Github Release Draft
  - Triggers only if a tag gets created
- `Release (Debug)`
  - By default disabled, but can be triggered manually for branches with the prefix `^(ci)/.*`
  - Exports the container to Google Cloud Storage, Pushes the images to public build cache and create the Github Release **(TODO add configuration for public build cache for debug releases)**
- `Export <flavor>`
  - Exports the container of a flavor to Google Cloud Storage

Additional commands:

- `cancel_all_builds.sh`: Cancels all builds in the current project, not only the builds of this setup
- `run_triggers.sh <commitSha> <file-pattern>`: Runs a trigger for a commit if its config file can be found by the file-pattern. The config files are stored under triggers/flavor-config
- `setup.sh` can be also used for updating the setups

## Additional Notes:
- The trigger_ids and encrypted passwords are saved in .env, this directory is cruical for updating the setup. **Create a Backup**
- (TODO add script which restores the trigger_ids from a existing setup for the current config)
