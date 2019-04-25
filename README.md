# Build the Dockerfile to create image
docker build -t jupyter/deeptest .
# run jupyter notebook in docker image
docker run --rm -it -p 8888:8888 -v "(volume)/GitHub/docker-dl/Notebook":/home/jovyan/work jupyter/deeptest
