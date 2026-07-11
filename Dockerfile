FROM node:20-bookworm-slim AS python-base

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/python-setup

COPY requirements.txt ./

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

ENV PATH="/opt/venv/bin:${PATH}"


FROM python-base AS feature-builder

WORKDIR /build

COPY build-reference-index.py card_data.json ./
COPY reference_cards ./reference_cards

RUN python build-reference-index.py \
        card_data.json \
        reference_cards \
        reference_index


FROM python-base AS runtime

WORKDIR /app

COPY package.json package-lock.json ./

RUN npm ci --omit=dev \
    && npm cache clean --force

COPY server.js split-deck.js read-deck.js identify-cards.py card_data.json ./

COPY --from=feature-builder \
    /build/reference_index \
    /app/reference_index

RUN chown -R node:node /app

ENV NODE_ENV="production"
ENV PYTHONUNBUFFERED="1"
ENV PYTHON_BIN="python"
ENV PORT="7860"

# Avoid thread oversubscription on Render's small CPU instances.
ENV OMP_NUM_THREADS="1"
ENV OPENBLAS_NUM_THREADS="1"
ENV MKL_NUM_THREADS="1"
ENV NUMEXPR_NUM_THREADS="1"
ENV OPENCV_FOR_THREADS_NUM="1"
ENV UV_THREADPOOL_SIZE="2"
ENV MALLOC_ARENA_MAX="2"

# Keep expensive debug PNG generation off in production.
ENV PVZH_DEBUG="0"

# If a shortlist is uncertain, preserve quality by scanning all
# cost-compatible cards exactly.
ENV PVZH_FULL_FALLBACK="0"

USER node

EXPOSE 7860

CMD ["node", "server.js"]
