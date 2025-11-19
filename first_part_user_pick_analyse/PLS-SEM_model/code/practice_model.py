import pandas as pd
import os
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from semopy import Model

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 读取数据
file_path = os.path.join(script_dir, '..', 'result', 'cleaned_survey_data.csv')
data = pd.read_csv(file_path)

# 查看列名确认数据正确性
print("📊 数据列名如下：")
print(data.columns.tolist())

# ========================
# 第一步：PU ~ RPA + TA + EA + TF
# ========================
X_pu = data[['RPA1', 'RPA2', 'RPA3', 'RPA4',
             'TA1', 'TA2', 'TA3',
             'EA1', 'EA2', 'EA3',
             'TF1', 'TF2', 'TF3', 'TF4', 'TF5']]
Y_pu = data[['PU1', 'PU2', 'PU3']]

pls_pu = PLSRegression(n_components=3)
pls_pu.fit(X_pu, Y_pu)
y_pred_pu = pls_pu.predict(X_pu)

print("\n📊 第一步：PU 模型系数矩阵（X 对 PU 的影响）：")
print(pls_pu.coef_)

print("\n📉 第一步：每个因变量的均方误差（MSE）：")
for i, y_col in enumerate(Y_pu.columns):
    mse = mean_squared_error(Y_pu.iloc[:, i], y_pred_pu[:, i])
    print(f"{y_col}: {mse:.4f}")

# ========================
# 第二步：PEU ~ TH + RPA + IS + SO
# ========================
X_peu = data[['TH1', 'TH2', 'TH3',
              'RPA1', 'RPA2', 'RPA3', 'RPA4',
              'IS1', 'IS2', 'IS3',
              'SO1', 'SO2', 'SO3']]
Y_peu = data[['PEU1', 'PEU2']]

pls_peu = PLSRegression(n_components=2)
pls_peu.fit(X_peu, Y_peu)
y_pred_peu = pls_peu.predict(X_peu)

print("\n📊 第二步：PEU 模型系数矩阵（X 对 PEU 的影响）：")
print(pls_peu.coef_)

print("\n📉 第二步：每个因变量的均方误差（MSE）：")
for i, y_col in enumerate(Y_peu.columns):
    mse = mean_squared_error(Y_peu.iloc[:, i], y_pred_peu[:, i])
    print(f"{y_col}: {mse:.4f}")

# ========================
# 第三步：BIU ~ PU
# ========================
X_biu = data[['PU1', 'PU2', 'PU3']]  # 使用 PU 的观测变量作为输入
Y_biu = data[['BIU1', 'BIU2', 'BIU3']]

pls_biu = PLSRegression(n_components=3)
pls_biu.fit(X_biu, Y_biu)
y_pred_biu = pls_biu.predict(X_biu)

print("\n📊 第三步：BIU 模型系数矩阵（X 对 BIU 的影响）：")
print(pls_biu.coef_)

print("\n📉 第三步：每个因变量的均方误差（MSE）：")
for i, y_col in enumerate(Y_biu.columns):
    mse = mean_squared_error(Y_biu.iloc[:, i], y_pred_biu[:, i])
    print(f"{y_col}: {mse:.4f}")

# ========================
# 第四步：结构方程模型 (SEM) 建模
# ========================
desc = """
# 测量模型
PU =~ PU1 + PU2 + PU3
PEU =~ PEU1 + PEU2
AP =~ AP1 + AP2 + AP3
TH =~ TH1 + TH2 + TH3
RPA =~ RPA1 + RPA2 + RPA3 + RPA4
SI =~ SI1 + SI2 + SI3
TA =~ TA1 + TA2 + TA3
ST =~ ST1 + ST2 + ST3
IS =~ IS1 + IS2 + IS3
EA =~ EA1 + EA2 + EA3
SO =~ SO1 + SO2 + SO3
TF =~ TF1 + TF2 + TF3 + TF4 + TF5
BIU =~ BIU1 + BIU2 + BIU3

# 结构模型
RPA ~ TH + TA + ST 
PU ~ RPA + TF + EA + AP 
PEU ~ RPA + TH + TA + IS  
BIU ~ PU + PEU + SO
"""

model = Model(desc)
model.fit(data)

# 输出原始参数估计
print("\n📊 模型参数估计结果（原始系数）：")
params = model.inspect()
print(params)
params_path = os.path.join(script_dir, '..', 'result', '模型参数估计结果（新原始系数）.csv')
params.to_csv(params_path, 
                     index=False, 
                     encoding='utf-8-sig')


