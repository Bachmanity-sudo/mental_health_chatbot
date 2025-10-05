#!/bin/bash
set -e

PORT="${PORT:-5000}"
APP_ENV_RAW="${APP_ENV:-}"

if [ -z "$APP_ENV_RAW" ]; then
  if [ -f "/.dockerenv" ] || [ -n "${AWS_EXECUTION_ENV:-}" ] || [ -n "${APP_RUNNER:-}" ] || \
     [ -n "${ECS_CONTAINER_METADATA_URI:-}" ] || [ -n "${ECS_CONTAINER_METADATA_URI_V4:-}" ]; then
    APP_ENV="cloud"
  else
    APP_ENV="local"
  fi
else
  APP_ENV="$(echo "$APP_ENV_RAW" | tr "[:upper:]" "[:lower:]")"
fi

if [ "$APP_ENV" = "cloud" ]; then
  WORKERS="${GUNICORN_WORKERS:-1}"
  THREADS="${GUNICORN_THREADS:-2}"
  exec gunicorn --bind "0.0.0.0:${PORT}" --workers "$WORKERS" --threads "$THREADS" app:app
else
  exec python3 app.py
fi
