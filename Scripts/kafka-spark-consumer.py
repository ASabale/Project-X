# spark_kafka_consumer.py
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, DoubleType, IntegerType, StringType
from pyspark.sql.functions import from_json, col

# Create SparkSession connected to your container
spark = SparkSession.builder \
    .appName("traffic_data_1") \
    .getOrCreate()

# Schema for the Kafka JSON payloads
schema = StructType() \
    .add("fraud_bool", IntegerType()) \
    .add("income", DoubleType()) \
    .add("name_email_similarity", DoubleType()) \
    .add("prev_address_months_count", IntegerType()) \
    .add("current_address_months_count", IntegerType()) \
    .add("customer_age", IntegerType()) \
    .add("days_since_request", DoubleType()) \
    .add("intended_balcon_amount", DoubleType()) \
    .add("payment_type", StringType()) \
    .add("zip_count_4w", IntegerType()) \
    .add("velocity_6h", DoubleType()) \
    .add("velocity_24h", DoubleType()) \
    .add("velocity_4w", DoubleType()) \
    .add("bank_branch_count_8w", IntegerType()) \
    .add("date_of_birth_distinct_emails_4w", IntegerType()) \
    .add("employment_status", StringType()) \
    .add("credit_risk_score", IntegerType()) \
    .add("email_is_free", IntegerType()) \
    .add("housing_status", StringType()) \
    .add("phone_home_valid", IntegerType()) \
    .add("phone_mobile_valid", IntegerType()) \
    .add("bank_months_count", IntegerType()) \
    .add("has_other_cards", IntegerType()) \
    .add("proposed_credit_limit", DoubleType()) \
    .add("foreign_request", IntegerType()) \
    .add("source", StringType()) \
    .add("session_length_in_minutes", DoubleType()) \
    .add("device_os", StringType()) \
    .add("keep_alive_session", IntegerType()) \
    .add("device_distinct_emails_8w", IntegerType()) \
    .add("device_fraud_count", IntegerType()) \
    .add("month", IntegerType())

# --- Auto-create Iceberg table if it doesn't exist ---
spark.sql("""
    CREATE TABLE IF NOT EXISTS data.db.traffic_data_1 (
        fraud_bool INT,
        income DOUBLE,
        name_email_similarity DOUBLE,
        prev_address_months_count INT,
        current_address_months_count INT,
        customer_age INT,
        days_since_request DOUBLE,
        intended_balcon_amount DOUBLE,
        payment_type STRING,
        zip_count_4w INT,
        velocity_6h DOUBLE,
        velocity_24h DOUBLE,
        velocity_4w DOUBLE,
        bank_branch_count_8w INT,
        date_of_birth_distinct_emails_4w INT,
        employment_status STRING,
        credit_risk_score INT,
        email_is_free INT,
        housing_status STRING,
        phone_home_valid INT,
        phone_mobile_valid INT,
        bank_months_count INT,
        has_other_cards INT,
        proposed_credit_limit DOUBLE,
        foreign_request INT,
        source STRING,
        session_length_in_minutes DOUBLE,
        device_os STRING,
        keep_alive_session INT,
        device_distinct_emails_8w INT,
        device_fraud_count INT,
        month INT
    ) USING iceberg
""")

# Read stream from Kafka topic
df_raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "traffic-data") \
    .option("startingOffsets", "earliest") \
    .load()

# Convert binary Kafka 'value' field to string, then parse JSON
df_parsed = df_raw.selectExpr("CAST(value AS STRING) as json_string") \
    .select(from_json(col("json_string"), schema).alias("data")) \
    .select("data.*")

# Output: Print parsed messages to console
query = df_parsed.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .start()

query = df_parsed.writeStream \
    .format("iceberg") \
    .outputMode("append") \
    .option("checkpointLocation", "/home/iceberg/warehouse/checkpoints/traffic_data_1") \
    .toTable("db.traffic_data_1")

query.awaitTermination()
