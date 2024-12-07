import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

# 加载数据
data = pd.read_csv("1/apple_news_data.csv")
data['date'] = pd.to_datetime(data['date']).dt.strftime('%m-%d-%y')
data['date'] = pd.to_datetime(data['date'], format='%m-%d-%y')

# 筛选日期范围
start_date = "2019-01-01"
end_date = "2024-12-31"
filtered_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

# 去重：按日期保留第一条记录
filtered_data = filtered_data.drop_duplicates(subset=['date'], keep='first')

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
model.eval()

# 情感分析函数：返回每个类别的概率
def analyze_sentiment(text):
    if pd.isna(text):
        return [0.0, 1.0, 0.0]  # 默认值：Neutral 的概率为 1
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1).squeeze().tolist()  # 转换为列表
    return probs  # 返回 [Negative, Neutral, Positive] 的概率

# 显式循环处理每一行
predicted_probabilities = []
for _, row in tqdm(filtered_data.iterrows(), total=len(filtered_data), desc="Analyzing sentiments"):
    probabilities = analyze_sentiment(row['content'])
    predicted_probabilities.append(probabilities)

# 将概率拆分为三列并添加到 DataFrame
filtered_data[['pred_negative', 'pred_neutral', 'pred_positive']] = pd.DataFrame(predicted_probabilities, index=filtered_data.index)

i = 0
sentiment_dict = {}
for date, group in tqdm(filtered_data.groupby('date'), desc="Building sentiment dictionary"):
    total_negative = group['pred_negative'].sum()
    total_neutral = group['pred_neutral'].sum()
    total_positive = group['pred_positive'].sum()
    sentiment_dict[str(date)] = [total_negative, total_neutral, total_positive]
    i += 1
    if not i % 200:
        print("Saving~~~")
        with open("sentiment_dict.json", "w") as f:
            json.dump(sentiment_dict, f, indent=4)

with open("sentiment_dict.json", "w") as f:
    json.dump(sentiment_dict, f, indent=4)