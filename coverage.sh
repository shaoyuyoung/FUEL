#!/bin/bash
# shell script to collect code coverage for fuel
python experiments/collect_code_coverage.py \
--tech fuel \
--lib pytorch \
--folder results/fuel/pytorch-native