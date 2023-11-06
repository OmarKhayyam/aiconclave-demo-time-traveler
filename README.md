# aiconclave-demo-time-traveler



## Getting started

Make sure that you have cloned this repository. You can do this using intrictions from GitLab. Note that this code has been tested in __ap-south-1__ only.

## Add your files

Once you have cloned this repository. Please follow the below steps to set this demo up in your AWS account.
in the directory where you have cloned this repository:

`ls`

You should see two directories: `code` and `container`.

Next, download the PyTorch model file from this [Google drive location](https://drive.google.com/file/d/1ZILUGnwMyhrSYXaWnpFwzASVFt7ZXSSb/view?usp=drive_link)

Place the above downloaded file at the same level as the directories `code` and `container`, so your ls should show `model.pt`, apart from `code` and `container`. __Note__: If you have a `model.tar.gz` from previous runs of these commands, make sure you have deleted it using the following command: `rm model.tar.gz`.

Next, run the following command, in the same order as given below:
`tar czvf model.tar.gz ./* `
`aws s3 cp model.tar.gz s3://<Amazon S3 bucketname>/model.tar.gz` (Your S3 bucket may be different, make changes here and to the Notebook below accordingly)
`cd container`
`./build_and_push.sh`

Make a note of the ECR image URL from the last command above.

Next, launch a Amazon SageMaker Notebook instance. __NOTE__, we want to launch a Notebook instance, not the Studio UI.

Upload the `inference-test-NB-Working.ipynb` notebook to the Notebook instance, use the `conda_pytorch_p310` kernel for this notebook.

Run the first 10 cells of this Notebook, one after another, you will have to upload a photograph of yourself or a sample photograph to use as input, to the S3 input location when you call `invoke_endpoint_async`.

