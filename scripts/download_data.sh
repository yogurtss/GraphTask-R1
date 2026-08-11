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
  kilt)
    mkdir -p "$DATA_ROOT/kilt/2019-08-01"
    curl -L --fail --retry 3 --continue-at - \
      -o "$DATA_ROOT/kilt/2019-08-01/kilt_knowledgesource.json" \
      http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json
    ;;
  ssp)
    REVISION=ce7a0dfbc862f923ad1668a471c409b2e023b73f
    mkdir -p "$DATA_ROOT/ssp/$REVISION"
    curl -L --fail --retry 3 \
      -o "$DATA_ROOT/ssp/$REVISION/test.jsonl" \
      "https://huggingface.co/datasets/Quark-LLM/SSP/resolve/$REVISION/test.jsonl"
    echo "871c7b7cdec2e090e8597ef26a9a973a46aad0830bb1e016679dddd748462f50  $DATA_ROOT/ssp/$REVISION/test.jsonl" \
      | sha256sum --check --strict
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
    echo "Usage: DATA_ROOT=data/raw $0 {kqapro|kilt|ssp|freebase|webqsp|cwq|grailqa}"
    ;;
esac
