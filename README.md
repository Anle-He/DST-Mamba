<<<<<<< HEAD
# Decomposed Spatio-Temporal Mamba for Long-Term Traffic Prediction

python main.py -m DSTMamba -t LTSF -d PEMS08 -cfg ../baselines/DSTMamba/LTSFConfig/PEMS08_IN96_OUT12.yaml
=======
python exp_main.py -m DSTMamba -t LTSF -d PEMS08 -cfg ../baselines/DSTMamba/LTSFConfig/PEMS08_IN96_OUT12.yaml

# For test-only scenarios
python exp_main.py -m DSTMamba -t LTSF -tf True -d PEMS08 -cfg ../baselines/DSTMamba/LTSFConfig/PEMS08_IN96_OUT12.yaml
>>>>>>> 27484ad130b8bdb14335b99fece4e96864018dd5
