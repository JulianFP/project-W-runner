FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git

# install whisper first, as it's a very large dependency (multiple GBs of data) which
# we don't want to re-download every time we change the code.
RUN pip install whisperx

COPY . .

RUN --mount=source=.git,target=.git,type=bind \
    pip install --no-cache-dir -e .

CMD ["python", "-m", "project_W_runner"]
