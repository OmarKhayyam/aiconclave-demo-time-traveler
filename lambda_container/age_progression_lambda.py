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
ENDPOINT_NAME='time-traveler-endpoint-9'
CONTENT_TYPE='image/jpeg'
ACCEPTED_OUTPUT_FORMAT='application/x-npy'
NUM_IMAGES=5 ## This also includes the original input image

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

### We are provided with the location of the input file.
### We can assume that the input file is accessible to us.
### We create new output files and send the pre-signed URLs back.
def handler(event,context):
	## Get the input image location
	s3location = event["input_image_location"]
	### Let us use a private bucket where we create the prefix using the base name
	### of the input file appended with the current time-stamp.
	timerightnow = getCurrentTimestamp() ## This is a string
	inputfile_basename = (event["input_image_location"].split('/')[-1]).split('.')[0]
	s3location_output = BASEBUCKET + '/output/' + inputfile_basename + '-' + timerightnow ## format: s3://rns-conclave-2023/output/<basename of input file>-<TIMESTAMP>
	destination_url_collection = [] ## An empty list for now
	BUCKET_NAME = s3location.split('/')[2]
	KEY = '/'.join(s3location.split('/')[3:])
	logger.info("S3 location received {}".format(s3location))
	logger.info("Identified S3 bucket: {}".format(BUCKET_NAME))
	logger.info("Identified object key: {}".format(KEY))

	## Checking if the object exists
	s3 = boto3.client('s3',region_name=REGION)
	try:
		logger.info("Checking if input data exists at S3 location.")
		response = s3.head_object(Bucket=BUCKET_NAME,Key=KEY)
	except botocore.exceptions.ClientError as err:
		logger.info("Please check your input data. Something is wrong!")
		return {'statusCode':err.response["Error"]["Code"],'headers': {'ContentType': 'application/json'}, 'body':json.dumps({'Message': 'Please check your input data. Something is wrong!'})}
	
	## So we have now checked the object, it exists, and we are ready to call the model
	sagemaker_runtime = boto3.client('sagemaker-runtime',region_name=REGION)
	logger.info("Object exists.")
	logger.info("Invoking the model with the given input data.")
	model_response = sagemaker_runtime.invoke_endpoint_async(
						EndpointName=ENDPOINT_NAME,
						ContentType=CONTENT_TYPE,
						Accept=ACCEPTED_OUTPUT_FORMAT,
						InputLocation=s3location,
						InvocationTimeoutSeconds=900)
	if model_response['ResponseMetadata']['HTTPStatusCode'] == 202:
		logger.info("Received response from model.")
		Output_S3Location= model_response['OutputLocation']
		logger.info("Output file: {}".format(Output_S3Location))
        ### Let us wait for the output file, give the model some time
		time.sleep(10)
	### Read the output file from the S3 location
	bucket = Output_S3Location.split('/')[2]
	s3_res = boto3.resource('s3')
	ky = "/".join(Output_S3Location.split('/')[3:])
	outputfiles_basename= s3location.split('/')[-1].split('.')[0]
	try:
		resp = s3_res.meta.client.download_file(bucket,ky,'/tmp/{}.npy'.format(outputfiles_basename))
	except botocore.exceptions.ClientError as e:
		logger.error("Something went wrong, Could not get the inference results file.")
		return {'statusCode':e.response["Error"]["Code"],'headers': {'ContentType': 'application/json'}, 'body':json.dumps({'Message': 'Something went wrong, Could not get the inference results file.'})}
	results = np.load('/tmp/{}.npy'.format(outputfiles_basename))
	logger.info("Input data processed. Proceeding to create individual images.")
	### Now that we have our image results, we need to send these
	### images to the specified output location where the client
	### expects the images to be present.
	imgnum = 0
	while imgnum < NUM_IMAGES:
		img = getSpecificImage(imgnum,results)			
		img.save('/tmp/{}_{}.jpg'.format(outputfiles_basename,str(imgnum)))
		imgnum += 1
	logger.info("Individual files created. Uploading to Amazon S3.")
	imgnum = 0
	while imgnum < NUM_IMAGES:
		object_key = '{}_{}.jpg'.format(outputfiles_basename,str(imgnum))
		s3.upload_file('/tmp/'+ object_key,s3location_output.split('/')[2],'/'.join(s3location_output.split('/')[3:]) + '/' + object_key)
		destination_url_collection.append(s3location_output + '/' + object_key)
		imgnum += 1
	logger.info("Output files uploaded to {}/".format(s3location_output))

	### Generate pre-signed URLs
	ps_url_list = getPreSignedURLs(destination_url_collection)

	### If we are here, we are done. Let us return success back.
	return {'statusCode': 200, 'headers': {'ContentType': 'application/json'}, 'body': {'PreSignedURLs': json.dumps(ps_url_list)}}

################################################
### THIS IS THE DRIVER CODE FOR TESTING ONLY ###
################################################

if __name__ == '__main__':
	event = {'input_image_location':'s3://rns-conclave-2023/input/Test-Picture-Of-Self.jpg'}
	context = {'variable': 'NOTHING'}
	r = handler(event, context)
	pslist = json.loads(r['body']['PreSignedURLs'])
	for url in pslist:
		print("URL: \n{}\n".format(url))
