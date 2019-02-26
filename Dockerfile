FROM python:3

ADD lrcnModel.py /

RUN pip install numpy
RUN pip install pandas
RUN pip install matplotlib
RUN pip install os
RUN pip install pathlib
RUN pip install PIL
RUN pip install sklearn
RUN pip install keras

CMD [ "python", "./lrcnModel.py" ]
