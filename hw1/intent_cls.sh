#!/bin/bash
cd $(dirname $0)
cd intent
python test_intent.py --ckpt_path ../model/intent_model.ckpt

