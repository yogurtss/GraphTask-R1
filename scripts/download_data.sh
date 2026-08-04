#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data/raw}"
mkdir -p "$DATA_ROOT"

case "${1:-help}" in
  kqapro)
    command -v hf >/dev/null || { echo "Install huggingface_hub: pip install huggingface_hub"; exit 1; }
    hf download caobiao24/kqa_pro --repo-type dataset --local-dir "$DATA_ROOT/kqa_pro"
    echo "Verify attribution against the official KQA Pro CC BY-SA 4.0 release."
    ;;
  freebase)
    if [[ ! -d "$DATA_ROOT/freebase_setup/.git" ]]; then
      git clone --depth 1 https://github.com/dki-lab/Freebase-Setup.git "$DATA_ROOT/freebase_setup"
    fi
    echo "Follow $DATA_ROOT/freebase_setup/README.md; this wrapper does not download restricted dumps."
    ;;
  webqsp)
    echo "Download WebQSP from the official Microsoft/Freebase-Setup link."
    echo "Place JSON files under $DATA_ROOT/webqsp/"
    ;;
  cwq)
    echo "Download ComplexWebQuestions v1.1 from https://www.tau-nlp.sites.tau.ac.il/compwebq"
    echo "Place JSON files under $DATA_ROOT/complexwebquestions/"
    ;;
  grailqa)
    echo "Download GrailQA from https://dki-lab.github.io/GrailQA/"
    echo "Place files under $DATA_ROOT/grailqa/"
    ;;
  help|*)
    echo "Usage: DATA_ROOT=data/raw $0 {kqapro|freebase|webqsp|cwq|grailqa}"
    ;;
esac
