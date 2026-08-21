#!/bin/bash

export GPUSIM_CONFIG_JSON="${GPUSIM_CONFIG_JSON:-./gpusim_config.json}"

# Start gpusim http server
gpusim_run_server=$(jq -r .gpusim_run_server "${GPUSIM_CONFIG_JSON}")

if [[ "${gpusim_run_server}" == "true" ]]; then
  rm -f "${TMPDIR}/gpusimilarity"

  gpusim_folder_path=$(jq -r .gpusim_folder_path "${GPUSIM_CONFIG_JSON}")
  gpusim_db_filepath=$(jq -r .gpusim_db_filepath "${GPUSIM_CONFIG_JSON}")
  gpusim_port=$(jq -r .gpusim_port "${GPUSIM_CONFIG_JSON}")
  gpusim_virtualenv=$(jq -r .gpusim_virtualenv "${GPUSIM_CONFIG_JSON}")

  gpusim_runstring=(
    "$gpusim_virtualenv/bin/python"
    "$gpusim_folder_path/python/gpusim_server.py"
    "$gpusim_db_filepath"
    "--hostname" "localhost"
    "--port" "$gpusim_port"
    "--http_interface"
  )

  "${gpusim_runstring[@]}" 2>/dev/null &
  gpusim_expl_server_pid=$!

  # Give the server a moment to start
  sleep 1

  if kill -0 "$gpusim_expl_server_pid" 2>/dev/null; then
    echo "gpusimilarity exploration/exploitation server is running (PID: $gpusim_expl_server_pid)"
  else
    echo "gpusimilarity exploration/exploitation server failed to start."
    exit 1
  fi
fi
