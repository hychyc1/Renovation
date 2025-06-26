import os
import pandas as pd
import subprocess

# Correct checkpoints for each handle (trimmed paths)
checkpoints = {
    "cp": "saved_model/ckpt_cp_gnn_1155.67.pt",
    "cy": "saved_model/ckpt_cy_gnn_1960.76.pt",
    "dx": "saved_model/ckpt_dx_gnn_65.73.pt",
    "fs": "saved_model/ckpt_fs_gnn_267.72.pt",
    "ft": "saved_model/ckpt_ft_gnn_1489.30.pt",
    "hd": "saved_model/ckpt_hd_gnn_856.66.pt",
    "rem": "saved_model/ckpt_rem_gnn_56.71.pt",
    "sy": "saved_model/ckpt_sy_gnn_31.58.pt",
    "tz": "saved_model/ckpt_tz_gnn_52.62.pt"
}

# Handle-to-district mapping
handle_to_district = {
    "cp": "昌平区",
    "cy": "朝阳区",
    "dx": "大兴区",
    "fs": "房山区",
    "ft": "丰台区",
    "hd": "海淀区",
    "rem": "剩余五区", 
    "sy": "顺义区",
    "tz": "通州区"
}

# handle_to_district = {
#     "cp": "昌平区"
# }


# Final DataFrame to hold all results
final_df = pd.DataFrame()

# Iterate through each mapping
area = None
for handle, district in handle_to_district.items():
    checkpoint = checkpoints[handle]
    config_path = f"cfg/cfg_normal_gnn_3.yaml"
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
final_df.to_csv('baseline/district.csv')