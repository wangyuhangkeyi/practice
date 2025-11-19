from pathlib import Path
import pandas as pd
import numpy as np

script_dir = Path(__file__).resolve().parent
data_dir = script_dir / 'Data'

# 1. 读取文件（不自动解析时间列）
input_path = data_dir / 'dwv_order_make_haikou_1.txt'
output_parquet = data_dir / 'processed_haikou.parquet'

df = pd.read_csv(
    input_path,
    sep='\t',
    encoding='utf-8',
    dtype={
        'order_id': 'int64',
        'product_id': 'int32',
        'city_id': 'int16',
        'passenger_count': 'int8',
        'pre_total_fee': 'float32',
        'start_dest_distance': 'float32'
    },
    na_values=['', 'NULL', 'NA', 'null', '0000-00-00 00:00:00']  # 显式标记无效时间
)

# 2. 手动转换时间列（强制处理无效值）
for col in ['arrive_time', 'departure_time']:
    df[col] = pd.to_datetime(df[col], errors='coerce')  # 无效值转为NaT

# 3. 计算行程时长（过滤无效时间）
valid_times = df['arrive_time'].notna() & df['departure_time'].notna()
df.loc[valid_times, 'trip_duration'] = (df.loc[valid_times, 'arrive_time'] - 
                                       df.loc[valid_times, 'departure_time']).dt.total_seconds()

# 4. 数据清洗
df = df[valid_times]  # 删除时间无效的行
df['passenger_count'] = df['passenger_count'].fillna(1).astype('int8')  # 填充缺失值

# 5. 添加时间特征
df['hour'] = df['departure_time'].dt.hour
df['day_of_week'] = df['departure_time'].dt.dayofweek

# 6. 保存为Parquet格式
output_parquet.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(output_parquet, index=False)

print(f"预处理完成！已保存为 {output_parquet}")
print(f"处理后的数据维度: {df.shape}")
print(f"数据类型检查:\n{df.dtypes}")