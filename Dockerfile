FROM python:3.12-slim

# llama-cpp-python CPU wheel (no compiler needed), then quantgap.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir llama-cpp-python --prefer-binary \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[llamacpp]"

ENV HF_HOME=/root/.cache/huggingface
VOLUME ["/root/.cache/huggingface", "/results"]
WORKDIR /results
ENTRYPOINT ["quantgap"]
CMD ["--help"]
