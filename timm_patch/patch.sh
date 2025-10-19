#!/bin/bash

python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
timm_path=/usr/local/lib/python3.9/site-packages/timm/models/

cp timm_patch/swin_transformer.py $timm_path
cp timm_patch/helpers.py $timm_path/layers/