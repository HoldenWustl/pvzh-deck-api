FROM node:20-bookworm-slim

# Install Python and the small system libraries needed by OpenCV/SciPy.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The official Node image already contains the non-root "node" user.
WORKDIR /home/node/app

# Install Node dependencies first so Docker can cache this layer.
COPY --chown=node:node package.json package-lock.json ./

USER node

RUN npm ci --omit=dev

# Create the Python virtual environment and install Python dependencies.
COPY --chown=node:node requirements.txt ./

RUN python3 -m venv /home/node/venv \
    && /home/node/venv/bin/pip install \
        --no-cache-dir \
        --upgrade pip \
    && /home/node/venv/bin/pip install \
        --no-cache-dir \
        -r requirements.txt

# Copy the recognition scripts, JSON data, and reference images.
COPY --chown=node:node . .

ENV PATH="/home/node/venv/bin:${PATH}"
ENV NODE_ENV="production"
ENV PYTHONUNBUFFERED="1"

# Render supplies PORT automatically. This fallback is for local use.
ENV PORT="7860"

EXPOSE 7860

CMD ["node", "server.js"]