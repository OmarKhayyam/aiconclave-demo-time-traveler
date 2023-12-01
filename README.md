# aiconclave-demo-time-traveler



## Getting started

Make sure that you have cloned this repository. You can do this using intrictions from GitLab. Note that this code has been tested in __ap-south-1__ only.

## Add your files

Once you have cloned this repository. Please follow the below steps to set this demo up in your AWS account.
In the directory where you have cloned this repository:


`ls`


You should see two directories: `code` and `container`.


Next, download the PyTorch model file from this [Google drive location](https://drive.google.com/file/d/1ZILUGnwMyhrSYXaWnpFwzASVFt7ZXSSb/view?usp=drive_link). This has been specifically kept at this location for this purpose. We are not training/fine-tuning this model. If you want to know more about this model you can find it in this paper [here](https://arxiv.org/pdf/2102.02754.pdf), additionally, we use Dlib for facial landmark detection, this article provides a good [intro to Dlib](https://towardsdatascience.com/facial-mapping-landmarks-with-dlib-python-160abcf7d672).


Place the above downloaded file at the same level as the directories `code` and `container`, so your ls should show `model.pt`, apart from `code` and `container` directories. __Note__: If you have a `model.tar.gz` from previous runs of these commands, make sure you have deleted it using the following command: `rm model.tar.gz`.


Next, run the following command, in the same order as given below:

`tar czvf model.tar.gz ./* `

`aws s3 cp model.tar.gz s3://<Amazon S3 bucketname>/model.tar.gz` **(Your S3 bucket may be different, make changes here and to the Notebook below accordingly)**

`cd container`

`./build_and_push.sh`

Make a note of the ECR image URL from the previous command.

Next, launch a Amazon SageMaker Notebook instance. __NOTE__, we want to launch a Notebook instance, not the Studio UI.

Upload the `inference-test-NB-Working.ipynb` notebook to the Notebook instance, use the `conda_pytorch_p310` kernel for this notebook.

Run the first 10 cells of this Notebook, one after another, you will have to upload a photograph of yourself or a sample photograph to use as input, to the S3 input location when you call `invoke_endpoint_async`.

__DO NOT__ run the update endpoint cell, instead, skip to the delete enpoint cell to clean up your infrastructure.
