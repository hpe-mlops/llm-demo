FROM python:3.8

WORKDIR /workspace

# app
ADD src/requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt

ADD src/app.py src/ezm.py src/theme.py /workspace/
ADD images/app-header.png /workspace/image/
COPY documents /workspace/documents/

# kubectl
RUN curl -LO https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl
RUN chmod +x ./kubectl
RUN mv ./kubectl /usr/local/bin

# debug
RUN apt-get update
RUN apt-get install vim -y

ENV HOME=/workspace

EXPOSE 80

CMD [ "python3" , "/workspace/app.py" ]
