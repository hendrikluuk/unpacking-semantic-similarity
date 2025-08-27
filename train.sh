#!/bin/bash
./train_projections.py --out_dim 50 --save_model &
./train_projections.py --out_dim 100 --save_model &
./train_projections.py --out_dim 200 --save_model &
./train_projections.py --out_dim 400 --save_model &
./train_projections.py --out_dim 800 --save_model &
./train_projections.py --out_dim 1600 --save_model &
./train_projections.py --out_dim 3200 --save_model &
