FROM python:3.11-bullseye as spark-base

# Install tools required by the OS
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      sudo \
      curl \
      vim \
      unzip \
      openjdk-11-jdk \
      build-essential \
      software-properties-common \
      ssh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Setup the directories for our Spark and Hadoop installations
ENV SPARK_HOME=${SPARK_HOME:-"/opt/spark"}
ENV HADOOP_HOME=${HADOOP_HOME:-"/opt/hadoop"}
ENV PYTHONPATH=$SPARK_HOME/python/:$SPARK_HOME/python/lib/py4j-0.10.9.5-src.zip:$PYTHONPATH

RUN mkdir -p ${HADOOP_HOME} && mkdir -p ${SPARK_HOME}
WORKDIR ${SPARK_HOME}

# Download and install Spark
ENV SPARK_VERSION=3.5.5

ARG SPARK_DEFAULTS_CONF=spark-defaults.conf

RUN curl https://dlcdn.apache.org/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz -o spark-${SPARK_VERSION}-bin-hadoop3.tgz \
 && tar xvzf spark-${SPARK_VERSION}-bin-hadoop3.tgz --directory /opt/spark --strip-components 1 \
 && rm -rf spark-${SPARK_VERSION}-bin-hadoop3.tgz



FROM spark-base as pyspark-base

# Install python deps
COPY requirements/requirements.txt .
RUN pip3 install -r requirements.txt



FROM pyspark-base as pyspark

# Setup Spark related environment variables
ENV PATH="/opt/spark/sbin:/opt/spark/bin:${PATH}"
ENV SPARK_MASTER="spark://spark-iceberg:7077"
ENV SPARK_MASTER_HOST spark-iceberg
ENV SPARK_MASTER_PORT 7077
ENV PYSPARK_PYTHON python3

# Copy the default configurations into $SPARK_HOME/conf
#COPY spark-defaults.conf "$SPARK_HOME/conf"
COPY ${SPARK_DEFAULTS_CONF} "$SPARK_HOME/conf/spark-defaults.conf"

RUN chmod u+x /opt/spark/sbin/* && \
    chmod u+x /opt/spark/bin/*

# Copy appropriate entrypoint script
COPY entrypoint.sh .

ENTRYPOINT ["./entrypoint.sh"]


FROM pyspark as spark-iceberg

ARG SPARK_MAJOR_VERSION=3.5
ARG ICEBERG_VERSION=1.4.3
ARG SPARK_VERSION=3.5.5
ARG SCALA_VERSION=2.12
ARG ICEBERG_SPARK_SCALA="iceberg-spark-runtime-${SPARK_MAJOR_VERSION}_2.12"
ARG JAR_PACKAGE="${ICEBERG_SPARK_SCALA}-${ICEBERG_VERSION}.jar"

# Download iceberg spark runtime
RUN curl https://repo1.maven.org/maven2/org/apache/iceberg/${ICEBERG_SPARK_SCALA}/${ICEBERG_VERSION}/${JAR_PACKAGE} -Lo /opt/spark/jars/${JAR_PACKAGE}

# Download Java AWS bundle
# RUN curl https://repo1.maven.org/maven2/software/amazon/awssdk/bundle/2.17.257/bundle-2.17.257.jar -Lo /opt/spark/jars/bundle-2.17.257.jar
RUN curl -s https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-aws-bundle/${ICEBERG_VERSION}/iceberg-aws-bundle-${ICEBERG_VERSION}.jar -Lo /opt/spark/jars/iceberg-aws-bundle-${ICEBERG_VERSION}.jar

# Add all Kafka + transitive dependencies
RUN curl -s -L https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.5/spark-sql-kafka-0-10_2.12-3.5.5.jar -o /opt/spark/jars/spark-sql-kafka.jar && \
    curl -s -L https://repo1.maven.org/maven2/org/apache/spark/spark-token-provider-kafka-0-10_2.12/3.5.5/spark-token-provider-kafka-0-10_2.12-3.5.5.jar -o /opt/spark/jars/spark-token-provider-kafka.jar && \
    curl -s -L https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/3.4.1/kafka-clients-3.4.1.jar -o /opt/spark/jars/kafka-clients.jar && \
    curl -s -L https://repo1.maven.org/maven2/org/lz4/lz4-java/1.8.0/lz4-java-1.8.0.jar -o /opt/spark/jars/lz4-java.jar && \
    curl -s -L https://repo1.maven.org/maven2/org/xerial/snappy/snappy-java/1.1.10.5/snappy-java-1.1.10.5.jar -o /opt/spark/jars/snappy-java.jar && \
    curl -s -L https://repo1.maven.org/maven2/org/slf4j/slf4j-api/2.0.7/slf4j-api-2.0.7.jar -o /opt/spark/jars/slf4j-api.jar && \
    curl -s -L https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-client-runtime/3.3.4/hadoop-client-runtime-3.3.4.jar -o /opt/spark/jars/hadoop-client-runtime.jar && \
    curl -s -L https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-client-api/3.3.4/hadoop-client-api-3.3.4.jar -o /opt/spark/jars/hadoop-client-api.jar && \
    curl -s -L https://repo1.maven.org/maven2/commons-logging/commons-logging/1.1.3/commons-logging-1.1.3.jar -o /opt/spark/jars/commons-logging.jar && \
    curl -s -L https://repo1.maven.org/maven2/com/google/code/findbugs/jsr305/3.0.0/jsr305-3.0.0.jar -o /opt/spark/jars/jsr305.jar && \
    curl -s -L https://repo1.maven.org/maven2/org/apache/commons/commons-pool2/2.11.1/commons-pool2-2.11.1.jar -o /opt/spark/jars/commons-pool2.jar


# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
 && unzip awscliv2.zip \
 && sudo ./aws/install \
 && rm awscliv2.zip \
 && rm -rf aws/


# Install PostgreSQL JDBC Driver
RUN curl "https://jdbc.postgresql.org/download/postgresql-42.6.0.jar" -o "postgresql-42.6.0.jar" \
    && mv postgresql-42.6.0.jar "${SPARK_HOME}/jars/postgresql-42.6.0.jar"

# Add iceberg spark runtime jar to IJava classpath
ENV IJAVA_CLASSPATH=/opt/spark/jars/*

RUN mkdir -p /home/iceberg/localwarehouse /home/iceberg/notebooks /home/iceberg/warehouse /home/iceberg/spark-events /home/iceberg

ARG jupyterlab_version=4.0.8

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev && \
    pip3 install --upgrade pip && \
    pip3 install wget jupyterlab==${jupyterlab_version}

# Add a notebook command
RUN echo '#! /bin/sh' >> /bin/notebook \
 && echo 'export PYSPARK_DRIVER_PYTHON=jupyter' >> /bin/notebook \
 && echo "export PYSPARK_DRIVER_PYTHON_OPTS=\"lab --notebook-dir=/home/iceberg/notebooks --ip='0.0.0.0' --NotebookApp.token='' --port=8888 --no-browser --allow-root\"" >> /bin/notebook \
# && echo 'pyspark --master local[*]' >> /bin/notebook \
 && echo 'pyspark' >> /bin/notebook \
 && chmod u+x /bin/notebook


ENTRYPOINT ["./entrypoint.sh"]
CMD ["notebook"]
