import geopandas as gpd
import numpy as np

# 1. Read and clean
villages = gpd.read_file('data/urban_villages.shp')
villages = villages.dropna()

# 2. Compute area and drop the old 'Area' field
villages['area'] = villages.geometry.area
villages = villages.drop(columns=['Area'])

# 3. Select top 360 by area
villages = (
    villages
    .sort_values('area', ascending=False)
    .head(360)
)

# 4. Randomly shuffle
villages = villages.sample(frac=1, random_state=3).reset_index(drop=True)

# 5. Drop the old assignment columns
villages = villages.drop(columns=['assign_row', 'assign_col'])

# 6. Add 'year' in groups of 30 (1–12 over 360 rows)
villages['批次'] = villages.index // 30 + 1

# 7. Add constant fields
villages['容积率'] = 2.2
villages['模式']   = '4:2:4'

# 8. (Optional) Reorder columns for clarity
villages = villages[['ID', '批次', '容积率', '模式', 'geometry']]

# Now `villages` has exactly 360 rows,
# with `year` = 1 for rows 0–29, 2 for 30–59, …, 12 for 330–359,
# plus your two constant columns.
villages.to_file('baseline/greedy2/greedy2.shp')
