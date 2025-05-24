#iceberg with minio s3 storage and pg catalog instructions
down-iceberg-minio:
	docker compose -f docker-compose.yml -f docker-compose-minio.yml down

start-iceberg-minio:
	make stop-iceberg-minio && docker compose -f docker-compose.yml -f docker-compose-minio.yml up

stop-iceberg-minio:
	docker compose -f docker-compose.yml -f docker-compose-minio.yml stop

run-iceberg-minio:
	make stop-iceberg-minio && docker compose -f docker-compose.yml -f docker-compose-minio.yml up -d

build-services-spark-iceberg-minio:
	docker compose -f docker-compose.yml -f docker-compose-minio.yml build minio-s3 spark-iceberg spark-worker spark-history-server --no-cache

build-iceberg-minio:
	make down-iceberg-minio && docker compose -f docker-compose.yml -f docker-compose-minio.yml build

clean-iceberg-minio:
	docker compose -f docker-compose.yml -f docker-compose-minio.yml down --rmi="all" --volumes
