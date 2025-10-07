docker run -it --rm --name vollex-nim \
  --runtime=nvidia \
  --gpus all \
  --shm-size=16G \
  -e NGC_API_KEY=$NGC_API_KEY \
  -v ~/.cache/nim:/opt/nim/.cache \
  -u $(id -u) \
  -p 8000:8000 \
  nvcr.io/nvidia/vollex-nim:latest
