FROM python:3.13-slim@sha256:57b29a7c050c4e679e4bea572e14b3965cf4b5839de2da49d55e49ffc4024f39

# terminus 系 Agent 在容器里要起 tmux 会话，提前装好（跑 agent 时不依赖联网装系统包）
RUN apt-get update && apt-get install -y --no-install-recommends tmux && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/ /app/src/
COPY data/ /app/data/

CMD ["bash"]
