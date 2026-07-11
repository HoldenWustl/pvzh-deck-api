FROM node:20-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

WORKDIR /home/user/app

COPY --chown=user:user package.json package-lock.json ./

USER user

RUN npm ci --omit=dev

COPY --chown=user:user requirements.txt ./

RUN python3 -m venv /home/user/venv \
    && /home/user/venv/bin/pip install \
        --no-cache-dir \
        --upgrade pip \
    && /home/user/venv/bin/pip install \
        --no-cache-dir \
        -r requirements.txt

COPY --chown=user:user . .

ENV PATH="/home/user/venv/bin:${PATH}"
ENV NODE_ENV="production"
ENV PYTHONUNBUFFERED="1"
ENV PORT="7860"

EXPOSE 7860

CMD ["node", "server.js"]