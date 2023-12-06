#!/usr/bin/python3

import io
import sys
import json
import time
import boto3
import logging
import botocore
import requests
import numpy as np
from PIL import Image
from datetime import datetime

logger = logging.getLogger()
stream_handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(stream_handler)

REGION='ap-south-1'
BASEBUCKET='s3://rns-conclave-2023'
OUTPUTBASEBUCKET='s3://rns-conclave-2023-output'
ENDPOINT_NAME='time-traveler-endpoint-9'
CONTENT_TYPE='image/jpeg'
ACCEPTED_OUTPUT_FORMAT='application/x-npy'
NUM_IMAGES=5 ## This also includes the original input image
DYNAMODBTABLE='AgeProcessingJob'

def getCurrentTimestamp():
	return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

def getPreSignedURLs(s3_url_list):
	s3_cl = boto3.client('s3')
	ps_list = [] ## An empty list
	expiration = 900 ## Expiration set to 15 mins.
	for s3url in s3_url_list:
		bucket_name = s3url.split('/')[2]
		object_name = '/'.join(s3url.split('/')[3:])
		try:
			response = s3_cl.generate_presigned_url('get_object',
														Params={'Bucket': bucket_name,
														'Key': object_name},
														ExpiresIn=expiration)
		except botocore.exceptions.ClientError as e:
			logging.error(e)
			return None
		ps_list.append(response)	
	return ps_list

def doesItExist(s3_url):
	bkt = s3_url.split('/')[2]
	ky = '/'.join(s3_url.split('/')[3:])
	s3cl = boto3.client('s3')
	resul = s3cl.list_objects_v2(Bucket=bkt,Prefix=ky,MaxKeys=1)
	print(resul)
	if resul['IsTruncated'] == False and len(resul['Contents']) == 1:
		return True
	else:
		return False

def getSpecificImage(image_number,images_array):
    '''image_numbers start from 0, 
    where image_number == 0 is the original, 
    photographed image'''
    num_images = (images_array.shape[1])/(images_array.shape[0])
    if num_images <= image_number:
        print("Invalid image id requested.")
        return 
    if image_number == 0:
        starting_point = 0
        end_point = (starting_point + 1)*1024
    else:
        starting_point = image_number*1024
        end_point = starting_point+1024
    return Image.fromarray(images_array[:,starting_point:end_point,:])

### We can assume that the input file is accessible to us.
### We create new output files and put them in the output S3 location
def handler(event,context):
	dynamodb = boto3.resource('dynamodb')
	age_processing_job_table = dynamodb.Table(DYNAMODBTABLE)

	## Get the input image location
	BUCKET_NAME = event["Records"][0]["s3"]["bucket"]["name"]
	KEY = event["Records"][0]["s3"]["object"]["key"]
	s3location = "s3://"+ BUCKET_NAME + '/' + KEY
	inputfile_basename = (s3location.split('/')[-1]).split('.')[0] ##This is also the RequestId

	### Record incoming job	
	age_processing_job_table.put_item(
		Item={
		'RequestId': inputfile_basename,
		'CompletionStatus': 0,
		}
	)

	s3location_output = OUTPUTBASEBUCKET + '/output' ## format: s3://rns-conclave-2023-output/output/<basename of input file>_[0-4]
	logger.info("S3 location received {}".format(s3location))
	logger.info("Identified S3 bucket: {}".format(BUCKET_NAME))
	logger.info("Identified object key: {}".format(KEY))

	## We are ready to call the model
	sagemaker_runtime = boto3.client('sagemaker-runtime',region_name=REGION)
	logger.info("Invoking the model with the given input data.")
	model_response = sagemaker_runtime.invoke_endpoint_async(
						EndpointName=ENDPOINT_NAME,
						ContentType=CONTENT_TYPE,
						Accept=ACCEPTED_OUTPUT_FORMAT,
						InputLocation=s3location,
						InvocationTimeoutSeconds=300)
	if model_response['ResponseMetadata']['HTTPStatusCode'] == 202:
		logger.info("Received response from model.")
		Output_S3Location= model_response['OutputLocation']
		logger.info("Output file: {}".format(Output_S3Location))
        ### Let us wait for the output file, give the model some time
		wait_time=0
		while not doesItExist(Output_S3Location):
			time.sleep(10)
			if wait_time == 6: # We wait 60 seconds for the job to finish
				## The files will not be processed, corrupt file
				age_processing_job_table.update_item(
					Key={
					'RequestId': inputfile_basename,
					},
					UpdateExpression='SET CompletionStatus = :val1',
					ExpressionAttributeValues={
						':val1': -1
					}
				)
				print("The file could not be found at this location: {}".format(Output_S3Location))
				exit(0)
			wait_time += 1

	### Read the output file from the S3 location
	bucket = Output_S3Location.split('/')[2]
	s3_res = boto3.resource('s3')
	ky = "/".join(Output_S3Location.split('/')[3:])
	outputfiles_basename= inputfile_basename
	try:
		resp = s3_res.meta.client.download_file(bucket,ky,'/tmp/{}.npy'.format(outputfiles_basename))
	except botocore.exceptions.ClientError as e:
		logger.error("Something went wrong, Could not get the inference results file.")
	results = np.load('/tmp/{}.npy'.format(outputfiles_basename))
	logger.info("Input data processed. Proceeding to create individual images.")
	### Now that we have our image results, we need to send these
	### images to the specified output location where the client
	### expects the images to be present.
	imgnum = 0
	while imgnum < NUM_IMAGES:
		img = getSpecificImage(imgnum,results)			
		object_key = '{}_{}.jpg'.format(outputfiles_basename,str(imgnum))
		img.save('/tmp/' + object_key) 
		s3_res.meta.client.upload_file('/tmp/'+ object_key,s3location_output.split('/')[2],'/'.join(s3location_output.split('/')[3:]) + '/' + object_key)
		imgnum += 1
	logger.info("Individual files created and uploaded to Amazon S3.")

	### Updating the record with completion status
	age_processing_job_table.update_item(
		Key={
		'RequestId': inputfile_basename,
		},
		UpdateExpression='SET CompletionStatus = :val1',
		ExpressionAttributeValues={
			':val1': 1
		}
	)

################################################
### THIS IS THE DRIVER CODE FOR TESTING ONLY ###
################################################
#
#if __name__ == '__main__':
#	event = {'input_image_location':'s3://rns-conclave-2023/input/Test-Picture-Of-Self.jpg'}
#	context = {'variable': 'NOTHING'}
#	r = handler(event, context)
#	pslist = json.loads(r['body']['PreSignedURLs'])
#	for url in pslist:
#		print("URL: \n{}\n".format(url))
################################################