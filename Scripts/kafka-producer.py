# csv_producer.py
import time

import pandas as pd
import json
from confluent_kafka import Producer

# Kafka config
conf = {
    'bootstrap.servers': 'localhost:9094'
}
producer = Producer(conf)


# Delivery callback
def delivery_report(err, msg):
    if err:
        print(f"❌ Delivery failed: {err}")
    else:
        print(f"✅ Sent to {msg.topic()} [{msg.partition()}]")


# Load CSV
df = pd.read_csv('data/data.csv')  # Adjust the path if needed

# Send each row as JSON
for index, row in df.iterrows():
    time.sleep(0.01)  # Simulate delay
    message = row.to_dict()
    json_value = json.dumps(message)
    producer.produce('traffic-data', value=json_value, callback=delivery_report)
    producer.poll(0)

producer.flush()
