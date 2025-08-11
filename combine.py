import os
import pandas as pd
import subprocess

# Correct checkpoints for each handle (trimmed paths)
# checkpoints = {
    # "cp": "saved_model/ckpt_cp_gnn_1155.67.pt",
    # "cy": "saved_model/ckpt_cy_gnn_1960.76.pt",
    # "dx": "saved_model/ckpt_dx_gnn_65.73.pt",
    # "fs": "saved_model/ckpt_fs_gnn_267.72.pt",
    # "ft": "saved_model/ckpt_ft_gnn_1489.30.pt",
    # "hd": "saved_model/ckpt_hd_gnn_856.66.pt",
    # "rem": "saved_model/ckpt_rem_gnn_56.71.pt",
    # "sy": "saved_model/ckpt_sy_gnn_31.58.pt",
    # "tz": "saved_model/ckpt_tz_gnn_52.62.pt"
# }

model_path = 'saved_model/selfish/'

checkpoints = {
'ckpt_cp_49.34.pt',
'ckpt_cy_27.40.pt',
'ckpt_dx_18.91.pt',
'ckpt_fs_36.37.pt',
'ckpt_ft_15.30.pt',
'ckpt_hd_13.44.pt',
'ckpt_rem_20.80.pt',
'ckpt_sy_114.80.pt',
'ckpt_tz_39.10.pt',
}

# Handle-to-district mapping
handle_to_district = {
    "cy": "朝阳区",
    # "cp": "昌平区",
    # "dx": "大兴区",
    # "fs": "房山区",
    # "ft": "丰台区",
    # "hd": "海淀区",
    # "rem": "剩余五区", 
    # "sy": "顺义区",
    # "tz": "通州区"
}

# handle_to_district = {
#     "cp": "昌平区"
# }


# Final DataFrame to hold all results
final_df = pd.DataFrame()

# Iterate through each mapping
area = None
for handle, district in handle_to_district.items():
    checkpoint = model_path + ([item for item in checkpoints if handle in item][0])
    config_path = f"cfg/cfg_normal_gnn.yaml"
    result_path = "inferred_plan/combining/plan.csv"
    
    # Run inference
    cmd = [
        "python", "infer.py",
        "--config", config_path,
        "--district", district,
        "--checkpoint", checkpoint,
        "--name", "combining"
    ]
    subprocess.run(cmd, check=True)
    
    # Load result and append to final DataFrame
    # info_path = "inferred_plan/combining/report.csv"
    # info = pd.read_csv(info_path)
    # area = info['AREA'] if area is None else (area + info['AREA'])

    if os.path.exists(result_path):
        df = pd.read_csv(result_path)
        final_df = pd.concat([final_df, df], ignore_index=True)
    else:
        print(f"Warning: {result_path} not found for {district}")

# Output final DataFrame
print(final_df)
print(area)
final_df.to_csv('inferred_plan/朝阳区/plan.csv')