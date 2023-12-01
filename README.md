# aiconclave-demo-time-traveler



## Getting started

Make sure that you have cloned this repository. You can do this using intrictions from GitLab. Note that this code has been tested in __ap-south-1__ only.

## Add your files

Once you have cloned this repository. Please follow the below steps to set this demo up in your AWS account.
In the directory where you have cloned this repository:


`ls`


You should see 3 directories: `code`, `lambda_container`, `images` and `container`.


Next, download the PyTorch model file from this [Google drive location](https://drive.google.com/file/d/1ZILUGnwMyhrSYXaWnpFwzASVFt7ZXSSb/view?usp=drive_link) to the current working directory. 


The name of the file is `model.pt`. You can use [gdown](https://pypi.org/project/gdown/) to directly download the file, [these instructions](https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive) might be useful. 


We are not training/fine-tuning this model. If you want to know more about this model you can find it in this paper [here](https://arxiv.org/pdf/2102.02754.pdf), additionally, we will be using Dlib for facial landmark detection, this article provides a good [intro to Dlib](https://towardsdatascience.com/facial-mapping-landmarks-with-dlib-python-160abcf7d672).


Place the above downloaded file at the same level as the directories `code`, `lambda_container`, `README.md` and `container` in the cloned repository, so your ls should show `model.pt`, apart from the previously mentioned directories. __Note__: If you have a `model.tar.gz` from previous runs of these commands, make sure you have deleted it using the following command: `rm model.tar.gz`.


Next, run the following command, in the same order as given below:

`tar czvf model.tar.gz ./code model.pt`

This will create the structure [described in this documentation](https://sagemaker.readthedocs.io/en/v2.32.1/frameworks/pytorch/using_pytorch.html#model-directory-structure). Looks like the below picture.

![alt text](./images/Pytorch_SageMaker_Inference_Directory_Structure.jpg "Inference bundle for PyTorch on Amazon SageMaker")


Once you have created the model artifacts i.e. model.tar.gz, upload it to your preferred Amazon S3 location. You can add additional prefixes if you like.


`aws s3 cp model.tar.gz s3://<Amazon S3 bucketname>/model.tar.gz` 

**NB: Your S3 bucket and key name may differ, but has to always end with `model.tar.gz`**


Now, change directory as shown below.


`cd container`


Execute the following command. You can change the name of the repository by making changes to the build_and_push.sh.


`./build_and_push.sh`


The previous command will create the ECR repo, and push the image to it for SageMaker to use later. Make a note of the ECR image URL from the previous command, you will need it later.


Now that we have our model artifacts and the container image ready. Let us deploy the model to a SageMaker endpoint.


#### Model deployment


You should do this from a Jupyter Notebook. The following code in the notebook will ensure you deploy the model for this demo.

`
import boto3

aws_region='ap-south-1'

role = get_execution_role()

sagemaker_client = boto3.client('sagemaker', region_name=aws_region)

sagemaker_role= get_execution_role()
`

Set up the bucket and object key for the model artifacts.

` #Create a variable w/ the model S3 URI
s3_bucket = 'my_demo_bucket' # Provide the name of your S3 bucket
bucket_prefix=''
model_s3_key = f"model.tar.gz"

#Specify S3 bucket w/ model
model_url = f"s3://{s3_bucket}/{model_s3_key}"`

You created your inference container a few steps back, set that up.

`container = "111111111111.dkr.ecr.ap-south-1.amazonaws.com/my_repo:latest"`

We will be going for an asynchronous endpoint, because we aren't sure how long our model might take.

`model_name = 'model-name'

#Create model
create_model_response = sagemaker_client.create_model(
    ModelName = model_name,
    ExecutionRoleArn = sagemaker_role,
    PrimaryContainer = {
        'Image': container,
        'ModelDataUrl': model_url,
        'Environment':{
            'TS_MAX_REQUEST_SIZE': '100000000',
            'TS_MAX_RESPONSE_SIZE': '100000000',
            'TS_DEFAULT_RESPONSE_TIMEOUT': '1000'
        },
    })`

Create an endpoint configuration. I have removed all the optional parts from the code below.

`s3_bucket_new = 'my-demo-bucket-output' ## output bucket
endpoint_name = 'my-demo-endpoint'
endpoint_config_name = "my-demo-config-name"

create_endpoint_config_response = sagemaker_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name, # You will specify this name in a CreateEndpoint request.
    # List of ProductionVariant objects, one for each model that you want to host at this endpoint.
    ProductionVariants=[
        {
            "VariantName": "variant1", # The name of the production variant.
            "ModelName": model_name, 
            "InstanceType": "ml.g4dn.xlarge", # Specify the compute instance type.
            "InitialInstanceCount": 1 # Number of instances to launch initially.
        }
    ],
    AsyncInferenceConfig={
        "OutputConfig": {
            # Location to upload response outputs when no location is provided in the request.
            "S3OutputPath": f"s3://{s3_bucket_new}/output",
        },
        "ClientConfig": {
            # (Optional) Specify the max number of inflight invocations per instance
            # If no value is provided, Amazon SageMaker will choose an optimal value for you
            "MaxConcurrentInvocationsPerInstance": 4
        }
    }
)
`

Create your endpoint.

`create_endpoint_response = sagemaker_client.create_endpoint(
                                            EndpointName=endpoint_name, 
                                            EndpointConfigName=endpoint_config_name) `

The endpoint creation might take some time, but it will eventually get done. Your model is now ready for inference requests.

Invoke the model using this code,

`# Create a low-level client representing Amazon SageMaker Runtime
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name="ap-south-1")

# Specify the location of the input. Here, a single JPEG file. Note that SageMaker should have access to the S3 location.
# Check the Sagemaker execution role to ensure it can access S3.
input_location = "s3://my-image-bucket/input/Test-Picture-Of-A-Face.jpg"

# After you deploy a model into production using SageMaker hosting 
# services, your client applications use this API to get inferences 
# from the model hosted at the specified endpoint.
response = sagemaker_runtime.invoke_endpoint_async(
                            EndpointName=endpoint_name, 
                            ContentType='image/jpeg',
                            Accept='application/x-npy',
                            InputLocation=input_location,
                            InvocationTimeoutSeconds=900)`


You will find the model output serialized at the location which you can find in the `response` to the `invoke_endpoint_async` call. You can find it in response["OutputLocation"]. You can isolate the S3 bucket (which you already have) and the key to download the inference results using the code below, replace with your own specific values.

`import botocore.exceptions

s3 = boto3.client('s3')
try:
    resp = s3.download_file(s3_bucket_new,'somefolder_prefix/some_object_key','local_filename')
except Exception as e:
    pprint(e)`


To test the model's inference capability, try the code below. This function displayes the specific image you want as the output of the cell where you call it. In the current setup and based on the current inference code, we get 5 images back, the first is the original image, and the rest from subscript 1 to 4 are part of the inference process.

`from PIL import Image
import numpy as np

results = np.load(f"local_filename")`

`def showSpecificImage(image_number,images_array):
    '''image_numbers start from 0, 
    where image_number == 0 is the original, 
    photographed image'''
    if int(images_array.shape[1]/images_array.shape[0]) <= image_number:
        print("Invalid image id requested.")
        return
    if image_number == 0:
        starting_point = 0
        end_point = (starting_point + 1)*1024
    else:
        starting_point = image_number*1024
        end_point = starting_point+1024
    img = Image.fromarray(images_array[:,starting_point:end_point,:])
    img.show()`

showSpecificImage(2,results)

All of the above code and a lot of experiments are present in the Jupyter Notebook available in this repository **inference-test-NB-Working.ipynb**.


**WIP**

Next, launch a Amazon SageMaker Notebook instance. __NOTE__, we want to launch a Notebook instance, not the Studio UI.

Upload the `inference-test-NB-Working.ipynb` notebook to the Notebook instance, use the `conda_pytorch_p310` kernel for this notebook.

Run the first 10 cells of this Notebook, one after another, you will have to upload a photograph of yourself or a sample photograph to use as input, to the S3 input location when you call `invoke_endpoint_async`.

__DO NOT__ run the update endpoint cell, instead, skip to the delete enpoint cell to clean up your infrastructure.
