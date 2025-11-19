import os
import pandas as pd
import jieba
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 📁 输入文件路径（请根据实际路径修改）
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, '..', 'result', 'analyzed_related_comments_with_features_and_likes.csv')
file_dir = os.path.dirname(file_path)

def plot_sentiment_pie(df):
    sentiment_counts = df['情感类别'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
    plt.title('微博情感分布')
    plt.axis('equal')  # 保证饼图为圆形
    plt.show()

def plot_top_keywords(df):
    keyword_columns = [col for col in df.columns if col.startswith('关键词_')]
    keyword_counts = df[keyword_columns].sum().sort_values(ascending=False)
    top_keywords_dict = keyword_counts.to_dict()

    keyword_df = pd.DataFrame(list(top_keywords_dict.items()), columns=['关键词', '出现次数'])
    plt.figure(figsize=(10, 6))
    sns.barplot(x='出现次数', y='关键词', data=keyword_df.sort_values(by='出现次数', ascending=False).head(20))
    plt.title('Top 20 高频积极关键词')
    plt.xlabel('出现次数')
    plt.ylabel('关键词')
    plt.tight_layout()
    plt.show()

def plot_topic_heatmap(df, num_topics=10):
    topic_columns = [f"主题_{i}" for i in range(num_topics)]
    topic_probs = df[topic_columns].head(20)  # 显示前20条微博的主题分布
    plt.figure(figsize=(12, 8))
    sns.heatmap(topic_probs, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("微博主题分布热力图（前20条）")
    plt.xlabel("主题编号")
    plt.ylabel("微博序号")
    plt.show()

def generate_wordcloud(df):
    all_words = " ".join([" ".join(jieba.cut(str(text))) for text in df['微博内容']])
    wordcloud = WordCloud(width=800, height=600, background_color='white', font_path='simhei.ttf').generate(all_words)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("微博内容词云图")
    plt.show()

def plot_sentiment_score_histogram(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df['情感得分'], bins=20, kde=True, color='skyblue')
    plt.title('情感得分分布直方图')
    plt.xlabel('情感得分')
    plt.ylabel('微博数量')
    plt.grid(True)
    plt.show()

def visualize_analysis(df):
    print("📊 正在生成微博情感分布饼图...")
    plot_sentiment_pie(df)

    print("📊 正在生成高频关键词柱状图...")
    plot_top_keywords(df)

    print("📊 正在生成LDA主题分布热力图...")
    plot_topic_heatmap(df)

    print("📊 正在生成词云图...")
    generate_wordcloud(df)

    print("📊 正在生成情感得分分布直方图...")
    plot_sentiment_score_histogram(df)

if __name__ == "__main__":
    # ✅ 加载已处理的微博数据
    print("🔄 加载已处理的微博数据...")
    df = pd.read_csv(file_path, encoding='utf-8-sig')

    # 如果没有列名，手动设置
    if '微博内容' not in df.columns:
        df.columns = ['微博内容', '与话题相关的评论', '与话题无关的评论']

    # 执行所有可视化操作
    visualize_analysis(df)