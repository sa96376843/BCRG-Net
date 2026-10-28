#!/usr/bin/env sh
# Run from any directory. Public research datasets; review source licenses before redistribution.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
DATA="$ROOT/data"
mkdir -p "$DATA/Kvasir-SEG" "$DATA/CVC-ClinicDB" "$DATA/kvasir-instrument/official"

# Kvasir-SEG: 1,000 image-mask pairs
curl -k -L --retry 3 https://datasets.simula.no/downloads/kvasir-seg.zip -o /tmp/kvasir-seg.zip
unzip -q -o /tmp/kvasir-seg.zip -d "$DATA"

# CVC-ClinicDB mirror used in this reproducibility package
curl -L --retry 3 https://www.kaggle.com/api/v1/datasets/download/balraj98/cvcclinicdb -o /tmp/cvcclinicdb.zip
unzip -q -o /tmp/cvcclinicdb.zip -d "$DATA/CVC-ClinicDB"

# Kvasir-Instrument: 590 image-mask pairs; official author mirror
curl -k -L --retry 3 https://www.kaggle.com/api/v1/datasets/download/debeshjha1/kvasirinstrument -o /tmp/kvasirinstrument.zip
unzip -q -o /tmp/kvasirinstrument.zip -d "$DATA/kvasir-instrument/official"

python3 "$ROOT/src/verify_data.py"
