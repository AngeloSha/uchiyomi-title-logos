# The CPU environment the studio scripts run in (the one these logos were made with had the same versions).
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 git \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu torchvision==0.23.0 \
 && pip install --no-cache-dir numpy==2.2.6 pillow==11.3.0 opencv-python-headless==4.12.0.88 einops==0.8.1 \
      transformers==4.51.3 timm==1.0.15 kornia==0.8.0 safetensors==0.5.3 hydra-core==1.3.2 iopath
# Hi-SAM (Apache-2.0), pinned to the revision Uchiyomi's title extractor uses. Weights go in /models.
RUN git clone https://github.com/ymy-k/Hi-SAM.git /opt/hi-sam && git -C /opt/hi-sam checkout 69009434d4dba5541f228d8f5acb0754c333d417 \
 && rm -rf /opt/hi-sam/.git /opt/hi-sam/pretrained_checkpoint && ln -s /models /opt/hi-sam/pretrained_checkpoint
ENV PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_FORCE_WEIGHTS_ONLY_LOAD=1
WORKDIR /studio
