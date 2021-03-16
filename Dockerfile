# Gene Symbol Classifier Dockerfile
# https://github.com/Ensembl/gene_symbol_classifier

FROM python:3.8

LABEL maintainer="William Stark <william@ebi.ac.uk>"

ENV \
    POETRY_HOME="/opt/poetry" \
    POETRY_VERSION=1.1.5 \
    POETRY_VIRTUALENVS_CREATE=false

ENV PATH="${POETRY_HOME}/bin:${PATH}"

# install Poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

# specify working directory
WORKDIR /app

RUN mkdir --parents --verbose /app/data

# copy project dependencies files
COPY \
    pyproject.toml \
    poetry.lock \
    /app/

RUN poetry install --no-dev

COPY \
    fully_connected_pipeline.py \
    pipeline_abstractions.py \
    dataset_generation.py \
    /app/

COPY \
    symbols_capitalization_mapping.pickle \
    /app/data/

VOLUME /app/checkpoints
VOLUME /app/sequences

ENTRYPOINT ["python", "fully_connected_pipeline.py"]
