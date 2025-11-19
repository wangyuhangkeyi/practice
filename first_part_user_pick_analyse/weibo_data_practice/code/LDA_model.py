import os
from pathlib import Path
import pandas as pd
import re
import jieba
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models

# -----------------------------
# 📁 设置中文字体支持（解决绘图中文乱码）
# -----------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------
# 🧠 手动定义 id2label / label2id（适配本地情感模型）
# -----------------------------
id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}

# -----------------------------
# 📁 本地模型路径（请根据实际情况修改）
# -----------------------------
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]
# 模型路径：优先环境变量，否则使用仓库 models/sentiment
default_model_path = project_root / 'models' / 'sentiment'
model_path = Path(os.environ.get('SENTIMENT_MODEL_PATH', str(default_model_path)))

# -----------------------------
# 🧠 加载本地 HuggingFace 中文情感分析模型
# -----------------------------
print("🔄 正在加载 HuggingFace 中文情感分析模型（本地）...")

try:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        local_files_only=True,
        id2label=id2label,
        label2id=label2id
    )
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

except Exception as e:
    print(f"❌ 加载模型失败：{e}")
    exit(1)

# -----------------------------
# 🧠 中文情感预测函数
# -----------------------------
def predict_sentiment(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return '中性', 0.0
    result = classifier(text, truncation=True, max_length=512)[0]
    label = result['label']
    score = result['score']
    sentiment_label = '正面' if label == 'positive' else '负面'
    return sentiment_label, score

# -----------------------------
# 🧾 中文分词函数
# -----------------------------
def chinese_tokenize(text):
    return [word for word in jieba.cut(str(text)) if len(word.strip()) > 1]

# -----------------------------
# 🧠 LDA 主题建模函数
# -----------------------------
def perform_lda(texts, num_topics=10):
    tokenized_texts = [chinese_tokenize(text) for text in texts]
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        alpha='auto',
        eta=0.01,
        passes=10
    )

    def get_topic_distribution(bow):
        topics = lda_model.get_document_topics(bow)
        topic_dist = [0] * num_topics
        for t_id, prob in topics:
            topic_dist[t_id] = prob
        return topic_dist

    return lda_model, [get_topic_distribution(bow) for bow in corpus]

# -----------------------------
# 🔑 TF-IDF 关键词提取
# -----------------------------
def extract_keywords_tfidf(texts, top_n=20):
    valid_texts = [str(text) for text in texts if pd.notna(text) and str(text).strip() != ""]
    if not valid_texts:
        print("⚠️ 没有有效文本可用于关键词提取")
        return []

    vectorizer = TfidfVectorizer(tokenizer=chinese_tokenize, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(valid_texts)
    feature_array = vectorizer.get_feature_names_out()
    tfidf_sorting = tfidf_matrix.toarray().sum(axis=0).argsort()[::-1]
    top_keywords = [feature_array[i] for i in tfidf_sorting[:top_n]]
    return top_keywords

# -----------------------------
# 🧹 去除评论前缀 ID 函数
# -----------------------------
def clean_comment_prefix(text):
    if not isinstance(text, str):
        return ""
    match = re.match(r'^[^：]+：(.+)$', text.strip())
    if match:
        return match.group(1).strip()
    else:
        return text.strip()

# -----------------------------
# 🧠 提取并清除点赞数
# -----------------------------
def extract_and_clean_likes(text):
    if not isinstance(text, str):
        return "", 1

    # 匹配各种格式的「点赞数」结构，并提取数字
    like_pattern = r'(?:点赞数[:：]\s*(\d+)|（点赞数[:：]\s*(\d+)）|【点赞数[:：]\s*(\d+)】)'

    like_match = re.search(like_pattern, text)

    if like_match:
        # 提取点赞数
        like_count = 1
        for group in like_match.groups():
            if group:
                like_count = int(group)
                break
        # 删除整个点赞结构
        clean_text = re.sub(like_pattern, '', text).strip()
        return clean_text, like_count
    else:
        return text.strip(), 1

# -----------------------------
# 📁 输入文件路径（请根据实际路径修改）
# -----------------------------
file_path = script_dir.parent / 'result' / 'processed_tweets_final.csv'
file_dir = file_path.parent

# ✅ 第一步：加载并清洗“与话题相关的评论”数据

df_original = pd.read_csv(file_path, on_bad_lines='skip', encoding='utf-8-sig')

if '微博内容' not in df_original.columns:
    if len(df_original.columns) == 4:
        df_original.columns = ['微博内容', '与话题相关的评论', '与话题无关的评论', '点赞数']
    elif len(df_original.columns) == 3:
        df_original.columns = ['微博内容', '与话题相关的评论', '与话题无关的评论']

# ✅ 保留原始微博内容和点赞数
original_weibo_df = df_original[['微博内容', '点赞数']].copy()
original_weibo_df['微博内容'] = original_weibo_df['微博内容'].astype(str).apply(clean_comment_prefix)
original_weibo_df.dropna(subset=['微博内容'], inplace=True)
original_weibo_df = original_weibo_df[original_weibo_df['微博内容'] != ""].reset_index(drop=True)

# ✅ 清洗“与话题相关的评论”列
related_comments = []
likes_list = []

for comment_block in df_original['与话题相关的评论'].dropna():
    # 按 \n 或 \\ 分割每条评论
    lines = re.split(r'\n|\\', comment_block)
    for line in lines:
        cleaned_line, like_count = extract_and_clean_likes(line)
        if cleaned_line:
            related_comments.append(cleaned_line)
            likes_list.append(like_count)

df_related_comments = pd.DataFrame({
    '微博内容': related_comments,
    '点赞数': likes_list
})

# ✅ 合并原始微博内容 和 相关评论
combined_df = pd.concat([original_weibo_df, df_related_comments], ignore_index=True)

# ✅ 删除无效内容（纯数字、括号等）
def is_invalid_content(text):
    text = str(text).strip()
    pattern = r'^[\d\s\)\]\}\{\}]*$|^$|^[（$].*[）$]$'
    return bool(re.match(pattern, text))

combined_df['微博内容'] = combined_df['微博内容'].str.strip()
invalid_mask = combined_df['微博内容'].apply(is_invalid_content)
combined_df = combined_df[~invalid_mask].reset_index(drop=True)

# ✅ 去重处理：保留点赞数高的记录
cleaned_rows = {}
for _, row in combined_df.iterrows():
    content = row['微博内容'].strip()
    likes = int(row['点赞数'])
    if content not in cleaned_rows:
        cleaned_rows[content] = row
    else:
        if likes > int(cleaned_rows[content]['点赞数']):
            cleaned_rows[content] = row

new_df = pd.DataFrame(list(cleaned_rows.values()))
new_df = new_df[['微博内容', '点赞数']]
new_df['点赞数'] = new_df['点赞数'].astype(int)

# ✅ 删除微博内容中的空格和括号
new_df['微博内容'] = new_df['微博内容'].apply(lambda x: re.sub(r'\s+|[\(\)]', '', x))

# ✅ 删除开头是“文字加：”和结尾是“L加不定长文字加的微博”的内容
def clean_specific_content(text):
    # 删除开头是“文字加：”的内容
    text = re.sub(r'^[^：]+：', '', text)
    # 删除结尾是“L加不定长文字加的微博”的内容
    text = re.sub(r'L[^的]*的微博$', '', text)
    return text.strip()

new_df['微博内容'] = new_df['微博内容'].apply(clean_specific_content)

# ✅ 保存中间清洗后的 CSV
cleaned_output_path = file_dir / 'cleaned_related_comments_with_likes.csv'
new_df.to_csv(cleaned_output_path, index=False, encoding='utf-8-sig')
print(f"✅ 已清洗并保存带点赞数的相关评论至：{cleaned_output_path}")

# ✅ 第三步：情感分析
sentiments = []
scores = []

for index, row in new_df.iterrows():
    text = row['微博内容']
    sentiment, score = predict_sentiment(text)
    sentiments.append(sentiment)
    scores.append(score)

new_df['情感类别'] = sentiments
new_df['情感得分'] = scores
new_df['情感类别'] = new_df['情感类别'].astype('category')

# ✅ 第四步：LDA 主题建模
texts = new_df['微博内容'].dropna().tolist()
lda_model, topic_distributions = perform_lda(texts, num_topics=10)

# 打印每个主题的关键词及其权重
for i in range(lda_model.num_topics):
    topic_keywords = lda_model.print_topic(i)  # 提取每个主题的前10个关键词及其权重
    print(f"主题 {i}: {topic_keywords}")

# 将主题分布添加到 DataFrame 中
topic_columns = [f"主题_{i}" for i in range(lda_model.num_topics)]
for idx, topic_dist in enumerate(topic_distributions):
    new_df.loc[idx, topic_columns] = topic_dist

# ✅ 第五步：关键词提取（TF-IDF）并过滤情感倾向为积极的关键词
def extract_positive_keywords_tfidf(texts, top_n=20, excluded_keywords=None):
    if excluded_keywords is None:
        excluded_keywords = set()
    valid_texts = [str(text) for text in texts if pd.notna(text) and str(text).strip() != ""]
    if not valid_texts:
        print("⚠️ 没有有效文本可用于关键词提取")
        return []

    vectorizer = TfidfVectorizer(tokenizer=chinese_tokenize, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(valid_texts)
    feature_array = vectorizer.get_feature_names_out()
    tfidf_sorting = tfidf_matrix.toarray().sum(axis=0).argsort()[::-1]
    top_keywords = [feature_array[i] for i in tfidf_sorting[:top_n]]

    # 过滤掉特定的关键词
    filtered_keywords = [keyword for keyword in top_keywords if keyword not in excluded_keywords]

    # 过滤出情感倾向为积极的关键词
    positive_keywords = set()  # 使用集合去重
    for keyword in filtered_keywords:
        sentiment, _ = predict_sentiment(keyword)
        if sentiment == '正面':
            positive_keywords.add(keyword)

    # 如果过滤后关键词数量不足，补充其他高频关键词
    if len(positive_keywords) < top_n:
        other_keywords = [feature_array[i] for i in tfidf_sorting if feature_array[i] not in excluded_keywords]
        for keyword in other_keywords:
            sentiment, _ = predict_sentiment(keyword)
            if sentiment == '正面' and keyword not in positive_keywords:
                positive_keywords.add(keyword)
            if len(positive_keywords) >= top_n:
                break

    # 将集合转换为列表并返回
    return list(positive_keywords)

# 定义需要排除的关键词
excluded_keywords = {"萝卜", "武汉", "特斯拉", "司机","香港","这个","还是","驾驶","2025","公司","百度","现在","现在"}

# 提取高频积极关键词
top_keywords = extract_positive_keywords_tfidf(new_df['微博内容'], top_n=20, excluded_keywords=excluded_keywords)
print("📌 高频积极关键词：", top_keywords)

# 在 DataFrame 中添加关键词列
for keyword in top_keywords:
    new_df[f"关键词_{keyword}"] = new_df['微博内容'].str.contains(keyword, case=False).fillna(0).astype(int)

# ✅ 第六步：生成词云图
all_words = " ".join([" ".join(chinese_tokenize(text)) for text in new_df['微博内容']])
wc = WordCloud(width=800, height=600, background_color='white', font_path='simhei.ttf').generate(all_words)

plt.figure(figsize=(10, 6))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.title("微博评论词云图")
plt.show()

# ✅ 第七步：计算情感得分的加权平均值
new_df['情感得分加权'] = new_df['情感得分'] * new_df['点赞数']
weighted_sentiment_score = new_df['情感得分加权'].sum() / new_df['点赞数'].sum()
print(f"📌 情感得分的加权平均值：{weighted_sentiment_score:.4f}")

# ✅ 第八步：保存最终结果到 CSV
output_file = file_dir / 'analyzed_related_comments_with_features_and_likes.csv'
new_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"✅ 所有分析完成，结果已保存至 {output_file}")