docker-build:
	docker build -t jts .

docker-run:
	docker run -p 5001:5001 jts