# Build the Dockerfile to create image
docker build -t jupyter/deeptest .
# Run jupyter notebook in docker image
docker run --rm -it -p 8888:8888 -v "(VOLUME)/GitHub/docker-dl/Notebook":/home/jovyan/work jupyter/deeptest
# Open Jupyter Notebook server
http://127.0.0.1:8888/?token=(TOKEN)
