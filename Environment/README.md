# Project-X

## Overview
This project is a containerized setup for working with Apache Iceberg, MinIO S3 storage, PostgreSQL catalog, and Apache Spark. It also includes Kafka and Kafka UI for message streaming and monitoring. The project is designed to simplify the development and testing of data processing workflows.

## Prerequisites
- Docker and Docker Compose installed on your system.

## Services
The project includes the following services:
- **PostgreSQL (pg-catalog):** Used as the catalog for Apache Iceberg.
- **MinIO S3:** Provides S3-compatible object storage.
- **Apache Spark:** For distributed data processing.
- **Kafka:** For message streaming.
- **Kafka UI:** A web-based interface for monitoring Kafka.

## Makefile Commands
The `Makefile` provides several commands to manage the services:

### Start and Stop Services
- `make start-iceberg-minio`: Stops any running services and starts the Iceberg and MinIO setup.
- `make stop-iceberg-minio`: Stops the running Iceberg and MinIO services.
- `make run-iceberg-minio`: Starts the services in detached mode.

### Build Services
- `make build-services-spark-iceberg-minio`: Builds the MinIO, Spark, and related services without using cache.
- `make build-iceberg-minio`: Stops and rebuilds all services.

### Clean Up
- `make clean-iceberg-minio`: Stops all services, removes images, and deletes volumes.

### Tear Down
- `make down-iceberg-minio`: Stops and removes all services.

## Docker Compose Configuration
The `docker-compose-minio.yml` file defines the following services:
- **pg-catalog:** PostgreSQL database for Iceberg.
- **minio-s3:** MinIO object storage.
- **minio-s3-init:** Initializes MinIO with predefined buckets.
- **spark-iceberg:** Apache Spark configured for Iceberg.
- **spark-worker:** Spark worker node.
- **spark-history-server:** Spark history server.
- **kafka:** Kafka broker for message streaming.
- **kafka-ui:** Web interface for Kafka.

## Volumes and Networks
- **Volumes:**
    - `minio-s3-data`: Persistent storage for MinIO.
    - `kafka_data`: Persistent storage for Kafka.
- **Network:**
    - `iceberg-net`: Shared network for all services.

## Usage
1. Clone the repository.
2. Navigate to the project directory.
3. Use the `Makefile` commands to start, stop, or build the services.

