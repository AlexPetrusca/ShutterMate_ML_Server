FROM python:3.6

# Set the working directory to /usr/src/app
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY assets /usr/src/app/assets
COPY temp /usr/src/app/temp
COPY VisionPredictEngine.py /usr/src/app
COPY requirements.txt /usr/src/app

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
RUN wget -O /usr/local/lib/python3.6/site-packages/fastai/weights.tgz http://files.fast.ai/models/weights.tgz
RUN tar xvfz /usr/local/lib/python3.6/site-packages/fastai/weights.tgz -C /usr/local/lib/python3.6/site-packages/fastai
RUN rm /usr/local/lib/python3.6/site-packages/fastai/weights.tgz
RUN chmod 777 assets/engine/stockfish_10_x64

# Run VisionPredictEngine.py when the container launches
CMD ["python", "VisionPredictEngine.py"]