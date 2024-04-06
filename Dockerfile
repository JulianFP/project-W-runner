# bookworm is based on debian, so we can use apt-get
FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.source=https://github.com/JulianFP/project-W-runner
LABEL org.opencontainers.image.description="project-W runner production image"
LABEL org.opencontainers.image.licenses=MIT

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git

# install whisper first, as it's a very large dependency (multiple GBs of data) which
# we don't want to re-download every time we change the code.
RUN pip install openai-whisper

COPY . .

RUN pip install .

CMD ["python", "-m", "project_W_runner"]