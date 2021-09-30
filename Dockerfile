# Gene Symbol Classifier Dockerfile
# https://github.com/Ensembl/gene_symbol_classifier

FROM python:3.9

LABEL maintainer="William Stark <william@ebi.ac.uk>"

# Poetry installation environment variables
ENV \
    POETRY_HOME="/opt/poetry" \
    POETRY_VERSION=1.1.10 \
    POETRY_VIRTUALENVS_CREATE=false

# add Poetry bin directory to PATH
ENV PATH="${POETRY_HOME}/bin:${PATH}"

# install Poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -

# specify working directory
WORKDIR /app

# copy project dependencies files
COPY \
    pyproject.toml \
    poetry.lock \
    /app/

# install project dependencies
RUN poetry install --no-dev

# create /app/data directory
RUN mkdir --verbose /app/data

# copy pipeline program files
COPY \
    gene_symbol_classifier.py \
    utils.py \
    dataset_generation.py \
    /app/

VOLUME /app/checkpoints
VOLUME /app/data

ENTRYPOINT ["python", "/app/gene_symbol_classifier.py"]
