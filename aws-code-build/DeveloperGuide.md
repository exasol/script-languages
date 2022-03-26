## Local Run

AWS Code Build provides docker containers and a small script to run single builds locally.
Note: This does not work for batch builds, only single builds.
Another limitation is, that artifacts will not be uploaded to S3.

### Requirements
- Docker
- AWS CLI setup
- GIT


### Setup

[Here](https://docs.aws.amazon.com/codebuild/latest/userguide/use-codebuild-agent.html) are detailed scripts for the installation. 
You will find a script under ./aws-code-build/run_local/get.sh which runs the installation automatically.

### Run

- Set the flavor in aws-code-build/run_local_run_local.env
- Run aws-code-build/run_local/run_local.sh



## AWS Run

- AWS CodeBuild has registered a WebHook in [script-languages](https://github.com/exasol/script-languages) and [script-languages-release](https://github.com/exasol/script-languages-release), which will trigger AWS builds automatically
- Access the AWS Console and go to "CodeBuild"
- On the left side select Build -> Build projects 
- Select the SLCCodeBuild_... or SLCReleaseCodeBuild_... project
- In the tag select "Batch History", there you will see all started batch builds for a commit.
- Select a batch build, and you can see all single builds (for each flavor) which belong to the batch build