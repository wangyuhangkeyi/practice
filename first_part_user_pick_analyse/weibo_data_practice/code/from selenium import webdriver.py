import os
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException, TimeoutException
import time
import csv
import random

# 解析 chromedriver 路径（优先读取环境变量，其次使用仓库 drivers 目录）
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]
default_driver = project_root / "drivers" / "chromedriver.exe"
chromedriver_path = Path(
    os.environ.get("CHROMEDRIVER_PATH", str(default_driver))
)
if not chromedriver_path.exists():
    raise FileNotFoundError(
        f"Chromedriver not found at {chromedriver_path}. "
        "Set CHROMEDRIVER_PATH or place the driver in drivers/."
    )

service = Service(str(chromedriver_path))
option = Options()
option.add_experimental_option("debuggerAddress", "127.0.0.1:9527")
option.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")

web = webdriver.Chrome(service=service, options=option)
web.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
    "source": """
    Object.defineProperty(navigator, 'webdriver', {
      get: () => undefined
    })
  """
})
web.implicitly_wait(10)  # 等待网页的加载时间

def fetch_comments(post):
    try:
        # 找到 card-act 元素
        card_act = post.find_element(By.CSS_SELECTOR, "div.card-act")
        
        # 找到评论按钮并点击
        try:
            expand_button = card_act.find_element(By.CSS_SELECTOR, "a[action-type='feed_list_comment']")
            web.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", expand_button)
            time.sleep(random.uniform(1, 2))
            
            # 尝试点击评论按钮
            max_attempts = 5
            attempts = 0
            while attempts < max_attempts:
                try:
                    expand_button.click()
                    break
                except Exception as e:
                    print(f"Attempt {attempts + 1} failed to click comment button: {e}")
                    time.sleep(random.uniform(1, 2))
                    attempts += 1
            
            if attempts == max_attempts:
                print("Failed to click comment button after multiple attempts.")
                return []
            
        except Exception as e:
            print(f"No expand comments button found or already expanded: {e}")
            return []
        
        # 查找评论列表容器
        comment_list_div = post.find_element(By.CSS_SELECTOR, "div[node-type='feed_list_repeat']")
        
        # 等待评论完全加载
        WebDriverWait(comment_list_div, 1).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.card-review.s-ptb10"))
        )
        
        # 提取评论和点赞数
        all_comments_with_likes = []

        comment_elements = comment_list_div.find_elements(By.CSS_SELECTOR, "div.card-review.s-ptb10")

        for comment in comment_elements:
            try:
                comment_text = comment.find_element(By.CSS_SELECTOR, "div.txt").text.strip()

                # 提取该条评论的点赞数，并做判断：有就+1，没有就设为1
                try:
                    like_span = comment.find_element(By.CSS_SELECTOR, "span.woo-like-count")
                    like_count_str = like_span.text.strip()
                    
                    if like_count_str.isdigit():
                        like_count = int(like_count_str) + 1
                    else:
                        like_count = 1
                except Exception as e:
                    like_count = 1  # 默认设为1

                # 将点赞数附加到评论内容后
                final_comment = f"{comment_text} (点赞数: {like_count})"
                all_comments_with_likes.append(final_comment)

            except Exception as e:
                print(f"Error extracting comment or like count: {e}")
                continue

        return all_comments_with_likes

    except Exception as e:
        print(f"Failed to fetch comments: {e}")
        return []

def search_and_extract_tweets(keyword, max_tweets=200):
    base_url = f"https://s.weibo.com/weibo?q={keyword}&tw=realtime&Refer=weibo_realtime"
    page_number = 1
    all_tweets_with_comments = []

    while len(all_tweets_with_comments) < max_tweets:
        url = f"{base_url}&page={page_number}"
        print(f"Loading URL: {url}")
        
        try:
            web.get(url)
        except Exception as e:
            print(f"Failed to load URL {url}: {e}")
            break
        
        try:
            # 等待页面上的某个元素加载完成
            WebDriverWait(web, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div[action-type='feed_list_item']"))
            )
        except Exception as e:
            print(f"Failed to load page {page_number}: {e}")
            break
        
        # 查找所有的 feed_list_item div 标签
        posts = web.find_elements(By.CSS_SELECTOR, "div[action-type='feed_list_item']")
        
        if not posts:
            print(f"No tweets found on page {page_number}. Stopping.")
            break
        
        for post in posts:
            tweet_info = {'tweet': '', 'comments': [], 'likes': 1}
            
            # 提取微博正文内容
            try:
                content_div = post.find_element(By.CSS_SELECTOR, "p.txt")
                if content_div:
                    tweet_text = content_div.text.strip()
                    tweet_info['tweet'] = tweet_text
            except StaleElementReferenceException:
                print("Tweet content div is stale. Re-finding it...")
                try:
                    content_div = post.find_element(By.CSS_SELECTOR, "p.txt")
                    tweet_text = content_div.text.strip()
                    tweet_info['tweet'] = tweet_text
                except Exception as e:
                    print(f"Failed to re-extract tweet text: {e}")
            except Exception as e:
                print(f"Failed to extract tweet text: {e}")

            # 提取微博正文的点赞数，并处理：有则 +1，无则默认设为 1
            try:
                like_element = post.find_element(By.CSS_SELECTOR, "span.woo-like-count")
                like_count_str = like_element.text.strip()

                if like_count_str.isdigit():
                    like_count = int(like_count_str) + 1  # 有就 +1
                else:
                    like_count = 1  # 没有就默认为 1
            except Exception as e:
                print(f"Failed to extract like count, setting default to 1: {e}")
                like_count = 1  # 出错也默认为 1

            tweet_info['likes'] = like_count
            print(f"Likes (after processing): {like_count}")

            # 获取评论
            try:
                comments = fetch_comments(post)
                tweet_info['comments'] = comments
            except Exception as e:
                print(f"Failed to fetch comments for tweet: {e}")
            
            if tweet_info['tweet']:
                print(f"Found tweet: {tweet_info['tweet']}")  # 打印微博内容
                print(f"Comments: {tweet_info['comments']}")   # 打印评论内容
                all_tweets_with_comments.append(tweet_info)
            
            # 检查是否已经达到最大微博数量
            if len(all_tweets_with_comments) >= max_tweets:
                print(f"Reached maximum tweets limit of {max_tweets}. Stopping.")
                break
        
        page_number += 1
        # 增加随机等待时间，避免频繁请求
        wait_time = random.uniform(3, 6)
        print(f"Waiting for {wait_time} seconds before loading the next page.")
        time.sleep(wait_time)  # 等待下一页加载

    return all_tweets_with_comments[:max_tweets]

script_dir = Path(__file__).resolve().parent
default_filename = script_dir.parent / 'result' / 'tweets_with_comments7.csv'

def save_to_csv(tweets, filename=default_filename):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['Tweet', 'Comments', 'Likes'])  # 添加 Likes 列

        for tweet in tweets:
            tweet_text = tweet.get('tweet', '')
            comments = '\n'.join(tweet.get('comments', []))
            likes = tweet.get('likes', 1)
            writer.writerow([tweet_text, comments, likes])

if __name__ == "__main__":
    keyword = "萝卜快跑"  # 替换为你要搜索的关键词
    
    try:
        tweets_with_comments = search_and_extract_tweets(keyword)
    except Exception as e:
        print(f"An error occurred during the scraping process: {e}")
    finally:
        save_to_csv(tweets_with_comments)
        print(f"Tweets and comments saved to {default_filename}")

    # 关闭浏览器
    web.quit()