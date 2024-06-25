# syntax=docker/dockerfile:1.4
FROM amazonlinux:2023

# Install pre-reqs
RUN yum update -y && yum install shadow-utils cmake g++ gcc python3 python3-pip -y
RUN pip3 --version
RUN python3 -m pip install --user --upgrade pip
RUN python3 -m pip install --user --upgrade setuptools
ENV PATH="/root/.local/bin:${PATH}"

# Create the user
ARG USERNAME=ec2-user
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME
RUN useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

WORKDIR /app

# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install
RUN aws --version

# Install AWS SDK and configure credentials
RUN pip3 install boto3
RUN mkdir -p /root/.aws && \
    echo '[default]\nregion = ${AWS_REGION}\naws_access_key_id = ${AWS_ACCESS_KEY_ID}\naws_secret_access_key = ${AWS_SECRET_ACCESS_KEY}\naws_session_token = ${AWS_SESSION_TOKEN}' > /root/.aws/credentials

# Copy files
COPY requirements.txt /app
COPY home.py /app
COPY config.yaml /app
COPY style.css /app
COPY image.png /app

RUN pip3 install -r requirements.txt

RUN chown $USERNAME:$USERNAME /app

USER $USER

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "home.py", "--server.port=8501", "--server.address=0.0.0.0"]
