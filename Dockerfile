FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy metadata only
COPY pyproject.toml README.md ./

# Install pip and the heavy Torch CPU library first (cached)
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# NOW copy the code
COPY src ./src
COPY configs ./configs

# Install the rest of the dependencies (numpy, ray, etc.) and your package
RUN pip install --no-cache-dir .

ENTRYPOINT ["serology-impute"]
