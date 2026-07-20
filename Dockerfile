FROM python:3.12-slim AS build

WORKDIR /app
COPY pyproject.toml README.md ./
COPY tokensense ./tokensense
RUN pip install --no-cache-dir --prefix=/install ".[server]"

FROM python:3.12-slim

RUN useradd --create-home --uid 1000 tokensense
COPY --from=build /install /usr/local

USER tokensense
WORKDIR /home/tokensense

ENV TOKENSENSE_PORT=8317
EXPOSE 8317

ENTRYPOINT ["tokensense"]
CMD ["serve"]
