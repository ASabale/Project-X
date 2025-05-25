## Using Spark with Docker

While the notebook server is running, you can use any of the following commands to interact with Spark:

- **Spark Shell**:  
  `docker exec -it spark-iceberg spark-shell`

- **Spark SQL**:  
  `docker exec -it spark-iceberg spark-sql`

- **PySpark**:  
  `docker exec -it spark-iceberg pyspark`

### Running the Kafka-Producer Script
make sure that you have confluent-kafka-python installed in your host machine. You can install it using pip:
```bash
pip install confluent-kafka
```
Once you have the required library installed, you can run the Kafka producer script to send messages to the Kafka topic. The script is designed to produce messages that can be consumed by the Spark consumer script.
simply run the kafka-producer script on host machine using python:
```bash
python kafka-producer.py
```


### Running the Kafka-Spark Consumer Script

1. Place the `kafka-spark-consumer.py` script in the mounted directory so that it is accessible to Spark. Verify its presence using the following command:  
   `docker exec -it spark-iceberg ls /home/iceberg/scripts/`

2. Submit the script to Spark using one of the following commands:

    - Basic submission:  
      `docker exec -it spark-iceberg spark-submit /home/iceberg/scripts/kafka-spark-consumer.py`

    - Advanced submission with custom configurations:
      ```bash
      docker exec -it spark-iceberg spark-submit \
        --master spark://spark-iceberg:7077 \
        --deploy-mode client \
        --executor-cores 2 \
        --total-executor-cores 2 \
        --executor-memory 2g \
        --conf spark.dynamicAllocation.enabled=false \
        /home/iceberg/scripts/kafka-spark-consumer.py
      ```