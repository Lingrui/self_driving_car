FROM tensorflow/tensorflow:1.6.0-gpu-py3

ADD requirments.txt /requirments.txt
RUN pip3 install -r /requirments.txt
RUN apt-get update && apt-get install -y wget vim cmake protobuf-compiler libgtk2.0 python3-tk git cmake python3-setuptools eog

# Clone the Semantic-Segmentation-Suite repository
RUN git clone https://github.com/GeorgeSeif/Semantic-Segmentation-Suite.git /sssuite && cd /sssuite && git checkout e17c527c3e40d6bb611ca1e3ca7d637f3235d4d1

ADD main.py /sssuite/main.py
ADD main_eval.py /sssuite/main_eval.py
ADD main_inference.py /sssuite/main_inference.py
ADD main_predict.py /sssuite/main_predict.py
ADD import_cityscape_resize.py /sssuite/import_cityscape_resize.py
WORKDIR /sssuite

