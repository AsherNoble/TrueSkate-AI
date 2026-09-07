#!/bin/bash
# Compatibility launcher.
exec bash "$(cd "$(dirname "$0")/../.." && pwd)/scripts/model1/run_model1_tonight.sh" "$@"
