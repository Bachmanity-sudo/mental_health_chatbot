FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip

# Copy requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Copy app
COPY . /app
WORKDIR /app

EXPOSE 5000
CMD ["python3", "app.py"]
