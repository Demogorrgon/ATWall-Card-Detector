# Base image
FROM python:3.10-slim AS python-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# builder-base is used to build dependencies
FROM python-base AS builder-base
RUN buildDeps="build-essential" \
    && apt-get update \
    && apt-get install --no-install-recommends -y \
        curl \
        netcat \
        python3-dev \
    && apt-get install -y --no-install-recommends $buildDeps \
    && rm -rf /var/lib/apt/lists/*

# install dependencies
WORKDIR $PYSETUP_PATH
COPY ./requirements.txt ./
RUN pip install -r ./requirements.txt

# 'development' stage installs all dev deps and can be used to develop code.
# For example using docker-compose to mount local volume under /app
FROM python-base as development
ENV FASTAPI_ENV=development

COPY --from=builder-base $PYSETUP_PATH $PYSETUP_PATH

WORKDIR /app
COPY . .
ENV PYTHONPATH=.

# Start app on 8000 port
EXPOSE 8000
CMD ["uvicorn", "--reload", "--host=0.0.0.0", "--port=8000", "app.main:create_app"]
