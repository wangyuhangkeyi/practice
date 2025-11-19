import pandas as pd
import numpy as np

# 1. 读取CSV文件（指定中文编码）
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..', '..')
input_path = os.path.join(script_dir, '..', 'POI_HaiKou.csv')
output_parquet = os.path.join(script_dir, 'Data', 'processed_poi_hainan.parquet')

df = pd.read_csv(
    input_path,
    encoding='utf-8-sig',  # 解决中文列名乱码
    dtype={
        '名称': 'string',    # 使用string类型存储文本
        '大类': 'category',  # 分类变量用category类型节省内存
        '中类': 'category',
        '经度': 'float32',
        '纬度': 'float32',
        '省份': 'category',
        '城市': 'category',
        '区域': 'category'
    }
)

# 2. 数据清洗
# 去除经纬度异常值（海南范围：经度108-112，纬度18-20）
# df = df[
#     (df['经度'].between(108, 112)) & 
#     (df['纬度'].between(18, 20))
# ]

# 3. 保存为Parquet格式（高效压缩）
df.to_parquet(
    output_parquet,
    index=False,
    compression='snappy'  # 使用snappy压缩
)

print(f"预处理完成！已保存为 {output_parquet}")
print(f"处理后的数据维度: {df.shape}")
print(f"数据类型检查:\n{df.dtypes}")
print(f"城市分布:\n{df['城市'].value_counts()}")