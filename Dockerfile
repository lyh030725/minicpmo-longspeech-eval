FROM runpod/pytorch:1.0.7-cu1290-torch291-ubuntu2404

WORKDIR /workspace/minicpmo-longspeech-eval
COPY . .
RUN bash scripts/setup_runpod.sh

ENV HF_HOME=/workspace/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD ["bash"]
