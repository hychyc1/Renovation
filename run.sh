python infer.py --checkpoint checkpoints/ckpt_normal_180_13813.50.pt --config cfg/cfg_shift1.yaml --name shift1
python infer.py --checkpoint checkpoints/ckpt_normal_180_13813.50.pt --config cfg/cfg_normal.yaml --name normal
python infer.py --checkpoint checkpoints/ckpt_normal_180_13813.50.pt --config cfg/cfg_shift2.yaml --name shift2
python infer.py --checkpoint checkpoints/ckpt_mone_160_6853.93.pt --config cfg/cfg_mone.yaml --name mone
python infer.py --checkpoint checkpoints/ckpt_poi_200_10795.08.pt --config cfg/cfg_poi.yaml --name poi
python infer.py --checkpoint checkpoints/ckpt_trans_200_39.43.pt --config cfg/cfg_trans.yaml --name trans