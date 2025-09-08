Renovation

Code for paper "Balancing Growth and Equity: Multi-Objective Optimization in Urban Renewal with Deep Reinforcement Learning".

Required package:
pytorch, pandas, geopands.

To train it globally:
python train.py

To train it for each district:
python train.py --district DISTRICT_NAME

To infer a plan:
python infer.py --district DISTRICT_NAME --checkpoint WEIGHT_PATH
