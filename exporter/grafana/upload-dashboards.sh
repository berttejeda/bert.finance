#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAFANA_DIR="${SCRIPT_DIR}/dashboards"

# --- Required environment variables ---
# GRAFANA_URL        - e.g. https://grafana.example.com
# GRAFANA_API_KEY    - Bearer token or service-account token
# INFLUXDB_HOST      - e.g. https://influxdb.example.com
# INFLUXDB_TOKEN     - InfluxDB API token
# INFLUXDB_ORG       - e.g. leh
# INFLUXDB_BUCKET    - e.g. stocks
# OLLAMA_ENDPOINT    - e.g. https://ollama.example.com
# OLLAMA_MODEL       - e.g. gemma-4-E2B-it-uncensored-Q8_0:latest

ENV_FILE="${SCRIPT_DIR}/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
else
  echo "WARN: ${ENV_FILE} not found, relying on exported environment variables."
fi

REQUIRED_VARS=(
  GRAFANA_URL
  GRAFANA_API_KEY
  INFLUXDB_HOST
  INFLUXDB_TOKEN
  INFLUXDB_ORG
  INFLUXDB_BUCKET
  OLLAMA_ENDPOINT
  OLLAMA_MODEL
)

missing=()
for var in "${REQUIRED_VARS[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    missing+=("$var")
  fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "ERROR: Missing required environment variables:"
  printf '  %s\n' "${missing[@]}"
  echo ""
  echo "Export them or add them to ${SCRIPT_DIR}/.env"
  exit 1
fi

# Only substitute our variables — leave Grafana template vars like
# \${datasource}, \${bucket}, \${ticker} untouched.
ENVSUBST_VARS='${INFLUXDB_HOST} ${INFLUXDB_TOKEN} ${INFLUXDB_ORG} ${INFLUXDB_BUCKET} ${OLLAMA_ENDPOINT} ${OLLAMA_MODEL}'

DASHBOARDS=(
  "dashboard.json"
  "dashboard-all.json"
)

upload_dashboard() {
  local file="$1"
  local path="${GRAFANA_DIR}/${file}"

  if [[ ! -f "$path" ]]; then
    echo "WARN: ${path} not found, skipping."
    return
  fi

  echo "Processing ${file} ..."

  # Run envsubst with explicit variable list
  rendered=$(envsubst "${ENVSUBST_VARS}" < "$path")

  # Wrap in the Grafana dashboard API payload
  payload=$(jq -n --argjson dashboard "${rendered}" \
    '{dashboard: $dashboard, overwrite: true}')

  # Upload via Grafana HTTP API
  http_code=$(curl -s -o /tmp/grafana-upload-response.json -w "%{http_code}" \
    -X POST "${GRAFANA_URL}/api/dashboards/db" \
    -H "Authorization: Bearer ${GRAFANA_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "${payload}")

  if [[ "$http_code" == "200" ]]; then
    uid=$(jq -r '.uid // "unknown"' /tmp/grafana-upload-response.json)
    echo "  OK  — uid=${uid}, url=${GRAFANA_URL}/d/${uid}"
  else
    echo "  FAIL — HTTP ${http_code}"
    jq '.' /tmp/grafana-upload-response.json 2>/dev/null || cat /tmp/grafana-upload-response.json
    return 1
  fi
}

echo "=== Uploading dashboards to ${GRAFANA_URL} ==="
echo ""

errors=0
for db in "${DASHBOARDS[@]}"; do
  if ! upload_dashboard "$db"; then
    ((errors++))
  fi
  echo ""
done

if [[ $errors -gt 0 ]]; then
  echo "Done with ${errors} error(s)."
  exit 1
else
  echo "Done — all dashboards uploaded."
fi
