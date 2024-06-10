FROM nvcr.io/nvidia/tritonserver:24.05-vllm-python-py3

# pip 업그레이드
RUN python3 -m pip install --upgrade pip

# blinker 패키지 설치
RUN pip install blinker

# 나머지 패키지 설치, blinker를 건드리지 않음
RUN pip install requests mlflow boto3 protobuf pillow markdown Mako kiwisolver jmespath itsdangerous importlib-metadata gunicorn greenlet graphql-core fonttools entrypoints cycler contourpy cachetools sqlalchemy scikit-learn pandas matplotlib graphql-relay gitdb Flask docker deprecated botocore s3transfer opentelemetry-api opentelemetry-sdk opentelemetry-semantic-conventions graphene gitpython alembic --ignore-installed blinker
RUN pip install Path

# 나머지 필요한 패키지 설치
COPY ./model.py /opt/tritonserver/backends/vllm/model.py
