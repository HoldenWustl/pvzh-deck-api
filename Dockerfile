FROM node:20-bookworm-slim

# Install Python and system dependencies.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Node dependencies as root during the image build.
COPY package.json package-lock.json ./

RUN npm ci --omit=dev \
    && npm cache clean --force

# Install Python dependencies into a virtual environment.
COPY requirements.txt ./

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install \
        --no-cache-dir \
        --upgrade pip \
    && /opt/venv/bin/pip install \
        --no-cache-dir \
        -r requirements.txt

# Copy scripts, data, and reference card images.
COPY . .

# Let the runtime Node user access the application files.
RUN chown -R node:node /app

ENV PATH="/opt/venv/bin:${PATH}"
ENV NODE_ENV="production"
ENV PYTHONUNBUFFERED="1"
ENV PORT="7860"

# Run the actual server as a non-root user.
USER node

EXPOSE 7860

CMD ["node", "server.js"]