FROM python:3.10.1-slim-bullseye AS builder

COPY poetry.lock pyproject.toml ./
RUN python -m pip install --no-cache-dir poetry==1.4.2 \
    && poetry export --without-hashes --without dev,test -f requirements.txt -o requirements.txt

FROM python:3.10.1-slim-bullseye

WORKDIR /app

COPY --from=builder requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY elforecast ./elforecast
COPY infer.py ./
COPY train.py ./

RUN poetry env info && \
    poetry shell