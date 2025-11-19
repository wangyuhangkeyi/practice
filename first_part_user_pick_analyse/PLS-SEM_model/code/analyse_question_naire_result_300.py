import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from statsmodels.formula.api import logit
from sklearn.metrics import roc_curve, roc_auc_score 

# 设置 Matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 获取当前脚本所在目录
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..', '..')

# 加载数据（请将问卷数据放在result目录下）
data_path = os.path.join(script_dir, '..', 'result', 'question_naire_result_309.csv')
if not os.path.exists(data_path):
    # 如果文件不存在，尝试从项目根目录查找
    data_path = os.path.join(project_root, 'question_naire_result_309.csv')
data = pd.read_csv(data_path)

# 查看数据的基本信息
print("原始数据的基本信息：")
print(data.info())
print("\n数据的前几行：")
print(data.head())

# 数据清洗
# 1. 删除无关列
columns_to_drop = ['序号', '提交答卷时间', '所用时间', '来源', '来源详情', '来自IP']
data = data.drop(columns=columns_to_drop)

# 2. 重命名列名
new_column_names = {
    '性别:': '性别',
    '年龄:': '年龄',
    '教育程度:': '教育程度',
    '月收入:': '月收入',
    '是否拥有私家车:': '是否拥有私家车',
    '是否使用过网约车:': '是否使用过网约车',
    '是否听说过萝卜快跑:': '是否听说过萝卜快跑',
    '如果城市交通中车辆全部更换为无人驾驶出租车后，城市交通拥堵问题将得到极大缓解。': '感知有用性_拥堵缓解',
    '如果城市交通中车辆全部更换为无人驾驶出租车后，车祸事故的发生率将会急剧降低。': '感知有用性_事故降低',
    '如果城市中车辆全部更换为无人驾驶出租车后，居民个性化、差异化的出行需求更容易得到满足。': '感知有用性_需求满足',
    '使用手机客户端预约一辆无人驾驶出租车的操作，我可以轻松掌握。': '感知易用性_预约操作',
    '与无人驾驶出租车之间进行指令交互（如修改目的地、中途临时停车等），对我非常简单。': '感知易用性_交互简单',
    '我愿意选择环保的出行方式。': '利他性偏好_环保',
    '我愿意为减少交通拥堵做出贡献。': '利他性偏好_减少拥堵',
    '我愿意尝试新技术以推动社会进步。': '利他性偏好_尝试新技术',
    '我能熟练使用网约车打车软件。': '网约车出行习惯_熟练使用',
    '使用网约车出行已经成为我日常出行的常用方式之一，我很熟悉也很习惯。': '网约车出行习惯_习惯使用',
    '去往停车位紧张的目的地时，我常常使用网约车出行。': '网约车出行习惯_停车位紧张',
    '网约车平台总是能较快地为我匹配到附近车辆。': '网约车平台感知可靠性_快速匹配',
    '在雨雪等极端天气下，网约车平台能够为我分配到车辆。': '网约车平台感知可靠性_极端天气',
    '网约车平台推荐的路线是由人工智能算法基于大数据实时分析得到的，我认为推荐路线一定比我或司机凭借经验的判断更可靠。': '网约车平台感知可靠性_路线可靠',
    '网约车平台所使用的导航，总是能够根据一些临时性的封路、事故造成的拥堵等突发路况及时调整。': '网约车平台感知可靠性_路况调整',
    '如果人们看到我使用网约车出行我将感觉自豪。': '社会影响_自豪感',
    '我尊敬或敬重的、可能会影响我行为的人鼓励我采用网约车出行。': '社会影响_他人鼓励',
    '对我很重要的人会认为我应该使用网约车出行。': '社会影响_他人期望',
    '我对新技术（如无人驾驶）感到焦虑。': '技术焦虑',
    '我担心无人驾驶出租车的可靠性。': '可靠性担忧',
    '我对无人驾驶出租车的安全性感到担忧。': '安全性担忧',
    '我对社会整体的信任水平较高。': '社会信任',
    '我相信政府会确保无人驾驶出租车的安全性。': '政府信任',
    '我相信科技公司会负责任地推广无人驾驶技术。': '科技公司信任',
    '在决定是否使用无人驾驶出租车之前，我会查阅大量相关信息。': '信息搜索_查阅信息',
    '我会通过社交媒体了解无人驾驶出租车的用户反馈。': '信息搜索_社交媒体',
    '我会参考专家对无人驾驶出租车的评价。': '信息搜索_专家评价',
    '我非常关注环境保护。': '环境意识',
    '我愿意选择更环保的出行方式。': '环境意识_环保出行',
    '我认为无人驾驶出租车有助于减少环境污染。': '环境意识_减少污染',
    '我对无人驾驶出租车的整体看法是积极的。': '情感倾向_积极看法',
    '我对无人驾驶出租车的技术进步感到兴奋。': '情感倾向_技术进步',
    '我对无人驾驶出租车的未来发展充满期待。': '情感倾向_未来发展',
    '我最关注无人驾驶出租车的安全性。': '关注主题_安全性',
    '我关心无人驾驶出租车对就业的影响。': '关注主题_就业影响',
    '我对无人驾驶出租车的市场垄断问题感到担忧。': '关注主题_市场垄断',
    '我对无人驾驶出租车的技术进步感到非常兴奋。': '关注主题_技术进步',
    '我对无人驾驶出租车的环保效益感兴趣。': '关注主题_环保效益',
    '我支持无人驾驶出租车这种未来新兴出行方式。': '使用意向_支持',
    '我将会频繁地使用无人驾驶出租车服务。': '使用意向_频繁使用',
    '我会考虑不再购买私家车，而采用无人驾驶出租车完成出行目的。': '使用意向_替代私家车'
}
data = data.rename(columns=new_column_names)


# 3. 将分类变量转换为数值变量
# 性别
data['性别'] = data['性别'].map({'男': 1, '女': 0})

# 是否拥有私家车
data['是否拥有私家车'] = data['是否拥有私家车'].map({'是': 1, '否': 0})

# 是否使用过网约车
data['是否使用过网约车'] = data['是否使用过网约车'].map({'是': 1, '否': 0})

# 是否听说过萝卜快跑
data['是否听说过萝卜快跑'] = data['是否听说过萝卜快跑'].map({'是': 1, '否': 0})

# 年龄
age_mapping = {
    '18-25岁': 1,
    '26-35岁': 2,
    '36-45岁': 3,
    '46-55岁': 4,
    '>55岁': 5
}
data['年龄'] = data['年龄'].map(age_mapping)

# 教育程度
education_mapping = {
    '高中/中专及以下': 1,
    '专科': 2,
    '本科': 3,
    '硕士及以上': 4
}
data['教育程度'] = data['教育程度'].map(education_mapping)

# 月收入
income_mapping = {
    '<3000元': 1,
    '3000-6000元': 2,
    '6000-9000元': 3,
    '9000-12000元': 4,
    '>12000元': 5
}
data['月收入'] = data['月收入'].map(income_mapping)


# 4. 处理缺失值
# 使用中位数填充数值型变量的缺失值
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

# 使用众数填充分类变量的缺失值
categorical_columns = data.select_dtypes(include=['object']).columns
for column in categorical_columns:
    data[column] = data[column].fillna(data[column].mode()[0])

# 5. 检查异常值
# 假设所有数值型变量的异常值为超出3倍标准差的值
for column in numeric_columns:
    mean = data[column].mean()
    std = data[column].std()
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# 定义原始中文列名和对应的英文缩写
column_mapping = {
    '感知有用性_拥堵缓解': 'PU1',
    '感知有用性_事故降低': 'PU2',
    '感知有用性_需求满足': 'PU3',
    '感知易用性_预约操作': 'PEU1',
    '感知易用性_交互简单': 'PEU2',
    '利他性偏好_环保': 'AP1',
    '利他性偏好_减少拥堵': 'AP2',
    '利他性偏好_尝试新技术': 'AP3',
    '网约车出行习惯_熟练使用':'TH1',
    '网约车出行习惯_习惯使用':'TH2',
    '网约车出行习惯_停车位紧张':'TH3',
    '网约车平台感知可靠性_快速匹配':'RPA1',
    '网约车平台感知可靠性_极端天气':'RPA2',
    '网约车平台感知可靠性_路线可靠':'RPA3',
    '网约车平台感知可靠性_路况调整':'RPA4',
    '社会影响_自豪感': 'SI1',
    '社会影响_他人鼓励': 'SI2',
    '社会影响_他人期望': 'SI3',
    '技术焦虑': 'TA1',
    '可靠性担忧': 'TA2',
    '安全性担忧': 'TA3',
    '社会信任': 'ST1',
    '政府信任': 'ST2',
    '科技公司信任': 'ST3',
    '信息搜索_查阅信息': 'IS1',
    '信息搜索_社交媒体': 'IS2',
    '信息搜索_专家评价': 'IS3',
    '环境意识': 'EA1',
    '环境意识_环保出行': 'EA2',
    '环境意识_减少污染': 'EA3',
    '情感倾向_积极看法': 'SO1',
    '情感倾向_技术进步': 'SO2',
    '情感倾向_未来发展': 'SO3',
    '关注主题_安全性': 'TF1',
    '关注主题_就业影响': 'TF2',
    '关注主题_市场垄断': 'TF3',
    '关注主题_技术进步': 'TF4',
    '关注主题_环保效益': 'TF5',
        '使用意向_支持': 'BIU1',
    '使用意向_频繁使用': 'BIU2',
    '使用意向_替代私家车': 'BIU3'
}

# 保留 column_mapping 中定义的列
selected_columns = list(column_mapping.keys())
data_filtered = data[selected_columns]

# 将列名替换为英文缩写
data_filtered.rename(columns=column_mapping, inplace=True)

#  6.保存为新的CSV文件
output_path = os.path.join(script_dir, '..', 'result', 'cleaned_survey_data.csv')
data_filtered.to_csv(output_path, 
                     index=False, 
                     encoding='utf-8-sig')

print(f"\n清洗后的数据已保存为 {output_path}")

# 数据分析
# 1. 描述性统计
print("\n描述性统计：")
print(data.describe())
describe_path = os.path.join(script_dir, '..', 'result', 'describe_data.csv')
data.describe().to_csv(describe_path, index=False, encoding='utf-8-sig')
# 计算描述性统计
descriptive_stats = data.describe()

# 提取均值
mean_values = descriptive_stats.loc['mean']

# 定义原始中文列名和对应的英文缩写
column_mapping = {
    '感知有用性_拥堵缓解': 'PU1',
    '感知有用性_事故降低': 'PU2',
    '感知有用性_需求满足': 'PU3',
    '感知易用性_预约操作': 'PEU1',
    '感知易用性_交互简单': 'PEU2',
    '利他性偏好_环保': 'AP1',
    '利他性偏好_减少拥堵': 'AP2',
    '利他性偏好_尝试新技术': 'AP3',
    '网约车出行习惯_熟练使用':'TH1',
    '网约车出行习惯_习惯使用':'TH2',
    '网约车出行习惯_停车位紧张':'TH3',
    '网约车平台感知可靠性_快速匹配':'RPA1',
    '网约车平台感知可靠性_极端天气':'RPA2',
    '网约车平台感知可靠性_路线可靠':'RPA3',
    '网约车平台感知可靠性_路况调整':'RPA4',
    '社会影响_自豪感': 'SI1',
    '社会影响_他人鼓励': 'SI2',
    '社会影响_他人期望': 'SI3',
    '技术焦虑': 'TA1',
    '可靠性担忧': 'TA2',
    '安全性担忧': 'TA3',
    '社会信任': 'ST1',
    '政府信任': 'ST2',
    '科技公司信任': 'ST3',
    '信息搜索_查阅信息': 'IS1',
    '信息搜索_社交媒体': 'IS2',
    '信息搜索_专家评价': 'IS3',
    '环境意识': 'EA1',
    '环境意识_环保出行': 'EA2',
    '环境意识_减少污染': 'EA3',
    '情感倾向_积极看法': 'SO1',
    '情感倾向_技术进步': 'SO2',
    '情感倾向_未来发展': 'SO3',
    '关注主题_安全性': 'TF1',
    '关注主题_就业影响': 'TF2',
    '关注主题_市场垄断': 'TF3',
    '关注主题_技术进步': 'TF4',
    '关注主题_环保效益': 'TF5',
        '使用意向_支持': 'BIU1',
    '使用意向_频繁使用': 'BIU2',
    '使用意向_替代私家车': 'BIU3'
}

# 筛选出感兴趣的列（使用原始中文列名）
columns_of_interest = list(column_mapping.keys())
mean_values = mean_values[columns_of_interest]

# 绘制条形图时使用英文缩写
plt.figure(figsize=(14, 8))  # 设置图像大小
sns.barplot(x=[column_mapping[col] for col in mean_values.index], y=mean_values.values)  # 使用英文缩写作为X轴标签

# 设置标题和轴标签
plt.title('用户选择相关变量的均值条形图', fontsize=16)  # 设置标题
plt.xlabel('变量', fontsize=14)  # 设置X轴标签
plt.ylabel('均值', fontsize=14)  # 设置Y轴标签

# 自动调整子图参数，以确保子图之间有足够空间
plt.tight_layout()

chart_path = os.path.join(script_dir, '..', 'result', 'user_choice_bar_chart.png')
plt.savefig(chart_path)  # 保存图像文件
plt.show()  # 显示图像
# 2. 相关性分析
data1 = data.rename(columns=column_mapping)

correlation_matrix = data1.corr()
print("\n相关性矩阵：")
print(correlation_matrix)
correlation_path = os.path.join(script_dir, '..', 'result', 'correlation_matrix_data1.csv')
correlation_matrix.to_csv(correlation_path, index=False, encoding='utf-8-sig')
# 替换相关性矩阵的列名为英文缩写

# 3. 因子分析
# 选择与用户选择意向相关的变量
variables = [
    '感知有用性_拥堵缓解', '感知有用性_事故降低', '感知有用性_需求满足',
    '感知易用性_预约操作', '感知易用性_交互简单',
    '利他性偏好_环保', '利他性偏好_减少拥堵', '利他性偏好_尝试新技术',
    '网约车出行习惯_熟练使用', '网约车出行习惯_习惯使用', '网约车出行习惯_停车位紧张',
    '网约车平台感知可靠性_快速匹配', '网约车平台感知可靠性_极端天气',
    '网约车平台感知可靠性_路线可靠', '网约车平台感知可靠性_路况调整',
    '社会影响_自豪感', '社会影响_他人鼓励', '社会影响_他人期望',
    '技术焦虑', '可靠性担忧', '安全性担忧',
    '社会信任', '政府信任', '科技公司信任',
    '信息搜索_查阅信息', '信息搜索_社交媒体', '信息搜索_专家评价',
    '环境意识', '环境意识_环保出行', '环境意识_减少污染',
    '情感倾向_积极看法', '情感倾向_技术进步', '情感倾向_未来发展',
    '关注主题_安全性', '关注主题_就业影响', '关注主题_市场垄断',
    '关注主题_技术进步', '关注主题_环保效益'
]

# 提取相关变量
factor_data = data[variables]

# 进行因子分析
fa = FactorAnalyzer(n_factors=3, rotation='varimax')
fa.fit(factor_data)

# 输出因子载荷
print("\n因子载荷：")
print(pd.DataFrame(fa.loadings_, index=variables))
factor_output_path = os.path.join(script_dir, '..', 'result', 'factor_loadings.csv')
# pd.DataFrame(fa.loadings_, index=variables).to_csv(factor_output_path, index=False, encoding='utf-8-sig')

# 4. 主成分分析（PCA）
pca = PCA(n_components=3)
pca_result = pca.fit_transform(factor_data)

# 输出主成分分析结果
print("\n主成分分析结果：")
print(pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3']))

# 5. Logistic回归分析
# 定义因变量和自变量
X = data[variables]
y = data['是否听说过萝卜快跑']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 训练Logistic回归模型
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# 预测
y_pred = logreg.predict(X_test)

# 评估模型
print("\nLogistic回归模型评估：")
print(classification_report(y_test, y_pred))

# 6. 随机森林分类器
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n随机森林分类器评估：")
print(classification_report(y_test, y_pred_rf))

# 7. 支持向量机（SVM）
svm = SVC(random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("\n支持向量机（SVM）评估：")
print(classification_report(y_test, y_pred_svm))

# 8. 绘制混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 9. 绘制ROC曲线
y_pred_proba = rf.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("\nROC AUC Score:", roc_auc)

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()