FROM public.ecr.aws/lambda/python:3.11

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Upgrade pip
RUN pip install --upgrade pip

# Install the specified packages
RUN pip install -r requirements.txt

# Copy function code
COPY age_progression_lambda.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "age_progression_lambda.handler" ]
