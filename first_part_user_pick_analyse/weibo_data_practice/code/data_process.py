import pandas as pd
import re
import csv
import os

# 📁 输入文件路径（请根据实际路径修改）
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, '..', 'result', 'tweets_with_comments.csv')

# 🔍 获取文件所在目录
file_dir = os.path.dirname(file_path)

# ✨ 定义清洗函数
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 删除成对出现的 #，保留中间的内容（如 #萝卜快跑# → 萝卜快跑）
    text = re.sub(r'#([^#\n\r]+?)#', r'\1', text)
    
    # 删除“展开”和“视频”
    text = text.replace("展开", "").replace("视频", "").replace("c","").replace("回复","")
    
    # 删除孤立的 #
    text = text.replace("#", "")
    
    return text.strip()

# 📄 使用 pandas 自动识别分隔符，并跳过错误行
df = pd.read_csv(file_path, on_bad_lines='skip')

# 检查必要列是否存在
required_columns = ['Tweet', 'Comments', 'Likes']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"❌ CSV 文件中缺少必要列 {required_columns}，请检查文件格式")

# ✅ 初始化列表用于存储清洗后的数据
cleaned_rows = []

# 🔧 第一步：统一评论格式（增强处理逻辑）
for index, row in df.iterrows():
    tweet = clean_text(row['Tweet'])  # 清洗微博内容
    comment_block = clean_text(row['Comments'])  # 清洗评论块
    likes = row['Likes']  # 获取点赞数

    if pd.isna(comment_block):
        cleaned_rows.append([tweet, "", likes])
        continue

    # 去除多余的空白字符
    comment_block = str(comment_block).strip()

    # 按行分割
    lines = [line.strip() for line in comment_block.split('\n')]

    unified_comments = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # 判断是否是一个可能的评论ID（不含冒号，非空）
        if line and ':' not in line and '：' not in line:

            # 看下一行是否是以冒号开头的内容
            if i + 1 < len(lines) and (lines[i+1].startswith(':') or lines[i+1].startswith('：')):
                content_line = lines[i+1][1:].strip()  # 去掉冒号
                unified_comments.append(f"{line}：{content_line}")
                i += 2  # 跳过这两行
            else:
                unified_comments.append(line)
                i += 1
        else:
            unified_comments.append(line)
            i += 1

    # 合并成字符串，每条评论一行
    unified_comments_str = '\n'.join(unified_comments)
    
    cleaned_rows.append([tweet, unified_comments_str, likes])

# 📥 保存中间清洗结果，方便人工检查格式是否正确
cleaned_df = pd.DataFrame(cleaned_rows, columns=['微博内容', '统一后的评论', '点赞数'])
cleaned_output_file = os.path.join(file_dir, 'cleaned_tweets.csv')
cleaned_df.to_csv(cleaned_output_file, index=False, encoding='utf-8-sig')
print(f"✅ 已保存清洗后的中间数据至 {cleaned_output_file}")

# 📊 统计原始数据中的微博数量和评论总数
total_weibo_count = len(df)
total_comment_count = df['Comments'].dropna().apply(lambda x: len(str(x).split('\n'))).sum()
print(f"\n📊 初步处理完成：")
print(f" - 原始微博总数：{total_weibo_count}")
print(f" - 原始评论总数：{total_comment_count}")

# 🧠 第二步：合并重复微博内容 + 分类评论 + 保留最大点赞数 + 评论去重优先保留带点赞数的评论
processed_data_dict = {}

# 🔍 定义判断是否与话题相关的函数
def is_related_to_topic(text, topic_keywords):
    text = text.lower()
    for keyword in topic_keywords:
        if keyword.lower() in text:
            return True
    return False

# 📋 定义关键词
topic_keywords = ['萝卜快跑', 'Robotaxi', '无人车', '自动驾驶出租车', '无人驾驶', '车', '它','驾','司','乘','少','多']

# 🔄 遍历清洗后的数据进行后续处理
for index, row in cleaned_df.iterrows():
    content = clean_text(row['微博内容'])  # 再次清洗微博内容
    comments = row['统一后的评论']
    likes = row['点赞数']

    if pd.isna(comments):
        continue

    comment_lines = comments.split('\n')

    # 如果是第一次遇到该微博内容，则直接初始化
    if content not in processed_data_dict:
        processed_data_dict[content] = {
            'related': {},  # key=(cid, ct)，value=(line, likes)
            'unrelated': {},
            'likes': likes
        }
    else:
        # 如果已经存在，尝试比较点赞数，保留较大的那个
        current_likes = processed_data_dict[content]['likes']
        try:
            new_likes_num = float(likes) if pd.notna(likes) else 0
            curr_likes_num = float(current_likes) if pd.notna(current_likes) else 0
            if new_likes_num > curr_likes_num:
                processed_data_dict[content]['likes'] = likes
        except:
            pass

    related_comments = processed_data_dict[content]['related']
    unrelated_comments = processed_data_dict[content]['unrelated']

    for line in comment_lines:
        if not line.strip():
            continue

        try:
            # 分割评论ID和内容
            parts = line.split('：', 1)
            if len(parts) < 2:
                continue  # 跳过不合法格式
            comment_id, raw_comment = parts
            comment_id = comment_id.strip()
            raw_comment = raw_comment.strip()

            # 提取点赞数
            like_match = re.search(r'\(点赞数\s*[:：]\s*(\d+)\)$', raw_comment)
            comment_likes = int(like_match.group(1)) if like_match else 0

            # 去掉点赞数字段后的内容
            clean_comment = re.sub(r'\s*\(\s*点赞数\s*[:：]\s*\d+\s*\)$', '', raw_comment).strip()

            full_key = (comment_id, clean_comment)

            # 判断是否相关
            is_related = len(clean_comment) > 2 and is_related_to_topic(clean_comment, topic_keywords)

            target_dict = related_comments if is_related else unrelated_comments

            if full_key in target_dict:
                existing_line, existing_likes = target_dict[full_key]

                # 只有当前评论有点赞数、且已有评论无点赞数，或点赞更高才替换
                if (comment_likes > 0 and existing_likes == 0) or (comment_likes > existing_likes):
                    target_dict[full_key] = (f"{comment_id}：{raw_comment}", comment_likes)
            else:
                # 第一次出现，正常加入
                target_dict[full_key] = (f"{comment_id}：{raw_comment}", comment_likes)

        except Exception as e:
            print(f"⚠️ 无法解析的评论行: {line} | 错误: {e}")

# ✅ 定义去重函数：保留含点赞数的评论
def deduplicate_comments(comments_list):
    """
    对评论列表进行去重，相同内容优先保留带点赞数的评论。
    :param comments_list: 原始评论列表（格式为 '用户名：评论内容 (点赞数: 123)'）
    :return: 去重后的评论列表
    """
    comment_dict = {}  # key: (comment_id, clean_content)，value: (full_line, like_count)

    for line in comments_list:
        if not line.strip():
            continue

        try:
            cid, raw_comment = line.split('：', 1)
            cid = cid.strip()
            raw_comment = raw_comment.strip()

            like_match = re.search(r'\(点赞数\s*[:：]\s*(\d+)\)$', raw_comment)
            likes = int(like_match.group(1)) if like_match else 0

            clean_comment = re.sub(r'\s*\(\s*点赞数\s*[:：]\s*\d+\s*\)$', '', raw_comment).strip()

            key = (cid, clean_comment)

            if key not in comment_dict:
                comment_dict[key] = (line, likes)
            else:
                existing_line, existing_likes = comment_dict[key]
                if (likes > 0 and existing_likes == 0) or (likes > existing_likes):
                    comment_dict[key] = (line, likes)

        except Exception as e:
            print(f"⚠️ 无法解析的评论行: {line} | 错误: {e}")

    return [v[0] for v in comment_dict.values()]

# 定义后处理函数：精细化去重
def post_process_comments(comments_block):
    """
    对评论块进行精细化去重：
    - 相同用户 + 相同评论内容视为重复（忽略是否带点赞数）
    - 若只有一条带点赞数，则保留带点赞数的那条
    - 若都带点赞数，则保留点赞数更高的那条
    :param comments_block: 多条评论组成的字符串，每行一条评论
    :return: 去重后的评论列表（字符串列表）
    """
    lines = [line.strip() for line in comments_block.split('\n') if line.strip()]
    seen = {}  # key: (comment_id, clean_content) -> value: (full_line, likes)

    for line in lines:
        try:
            cid, raw_comment = line.split('：', 1)
            cid = cid.strip()
            raw_comment = raw_comment.strip()

            # 提取点赞数
            like_match = re.search(r'\(点赞数\s*[:：]\s*(\d+)\)$', raw_comment)
            likes = int(like_match.group(1)) if like_match else 0

            # 清洗掉点赞字段作为 clean_content
            clean_comment = re.sub(r'\s*\(\s*点赞数\s*[:：]\s*\d+\s*\)$', '', raw_comment).strip()

            key = (cid, clean_comment)

            if key not in seen:
                seen[key] = (line, likes)
            else:
                existing_line, existing_likes = seen[key]

                # 只要当前这条有点赞且更高，就替换
                if (likes > 0 and existing_likes == 0) or (likes > existing_likes):
                    seen[key] = (line, likes)

        except Exception as e:
            print(f"⚠️ 后处理时无法解析评论行: {line} | 错误: {e}")

    return [v[0] for v in seen.values()]

# 📤 构建最终输出数据
processed_data = []

# 遍历所有合并后的微博内容
for content, data in processed_data_dict.items():
    # 第一阶段去重（基于 key: (cid, comment) + likes）
    related_lines = deduplicate_comments([line for line, _ in data['related'].values()])
    unrelated_lines = deduplicate_comments([line for line, _ in data['unrelated'].values()])

    # 将其转换为字符串块
    related_block = '\n'.join(related_lines)
    unrelated_block = '\n'.join(unrelated_lines)

    # 第二阶段去重（后处理，确保最终格式统一、彻底去重）
    final_related = post_process_comments(related_block)
    final_unrelated = post_process_comments(unrelated_block)

    # 转回字符串形式
    related_str = '\n'.join(final_related)
    unrelated_str = '\n'.join(final_unrelated)
    likes = data['likes']

    processed_data.append([content, related_str, unrelated_str, likes])

# 📊 统计处理后微博数量 和 总评论数（相关 + 无关）
processed_weibo_count = len(processed_data)

# 👇 计算所有评论总数
total_related_comments = sum(len([x for x in item[1].split('\n') if x.strip()]) for item in processed_data)
total_unrelated_comments = sum(len([x for x in item[2].split('\n') if x.strip()]) for item in processed_data)
total_processed_comments = total_related_comments + total_unrelated_comments

# ✅ 最终输出文件也保存到原始文件目录下
final_output_file = os.path.join(file_dir, 'processed_tweets_final.csv')
with open(final_output_file, mode='w', newline='', encoding='utf-8-sig', errors='ignore') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL, escapechar='\\')
    writer.writerow(['微博内容', '与话题相关的评论', '与话题无关的评论', '点赞数'])
    writer.writerows(processed_data)

print(f"\n📊 所有处理完成：")
print(f" - 处理后微博总数：{processed_weibo_count}")
print(f" - 处理后评论总数：{total_processed_comments}")
print(f"✅ 最终处理结果已保存至 {final_output_file}")