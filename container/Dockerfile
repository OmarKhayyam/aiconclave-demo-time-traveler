ARG REGION=ap-south-1

FROM 763104351884.dkr.ecr.$REGION.amazonaws.com/pytorch-inference:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker

ENV PATH="/opt/ml/code:${PATH}"

ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

ENV SAGEMAKER_PROGRAM inference.py

RUN git clone https://github.com/yuval-alaluf/SAM.git /opt/ml/code/SAM

RUN wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip

RUN unzip ninja-linux.zip -d /usr/local/bin/

RUN update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force

RUN cd /opt/ml/code/SAM && \
    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
    bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir dlib && \
    pip install --no-cache-dir nvgpu

RUN pip freeze
