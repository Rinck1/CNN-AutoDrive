#!/usr/bin/env bash
# MiniFlows Web Studio 一键部署（macOS / Linux）
set -e
cd "$(dirname "$0")"
python3 deploy_web.py "$@"
