FROM python:3

ADD lrcnModel.py /

CMD [ "python", "./lrcnModel.py" ]
