# NVIDIA NGC Configuration
# NGC_API_KEY='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'
# NGC_ORG_ID='0509588398571510'
# echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

export LOCAL_NIM_CACHE=~/.cache/nim
mkdir -p "$LOCAL_NIM_CACHE"
chmod 1777 $LOCAL_NIM_CACHE


export CONTAINER_NAME=nvcr.io/nim/black-forest-labs/flux.1-dev:1.1.0


docker run -it --rm --name=nim-server \
   --runtime=nvidia \
   --gpus='"device=0"' \
   -e NGC_API_KEY=$NGC_API_KEY \
   -e HF_TOKEN=$HF_TOKEN \
   -p 8000:8000 \
   -v "$LOCAL_NIM_CACHE:/opt/nim/.cache/" \
   $CONTAINER_NAME

