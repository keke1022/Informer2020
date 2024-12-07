import pandas as pd

file_path = "./data/us_stock_price/stock_market_price.csv"
data = pd.read_csv(file_path)

data['Date'] = pd.to_datetime(data['date'], format='%m-%d-%Y')  # 确保日期列是日期类型
data = data.sort_values(by='Date').reset_index(drop=True)  # 按日期升序排序

# 假设 df 是你的 DataFrame，length 是切片的长度
length = 6  # 你可以自行调整这个值

# 提取所有股票的价格列名和成交量列名
stock_price_columns = [col for col in data.columns if '_Price' in col]
stock_volume_columns = [col.replace('_Price', '_Vol.') for col in stock_price_columns]

# 初始化存储结果的列表
result = []

# 遍历每个股票价格列和对应的成交量列
for price_col, volume_col in zip(stock_price_columns, stock_volume_columns):
    for i in range(0, len(data) - length + 1, length):
        # 生成一个切片
        sliced_data = {
            'Stock': price_col,
            'Date': list(data['Date'].iloc[i:i+length]),
            'Prices': list(data[price_col].iloc[i:i+length]),
            'Volumes': list(data[volume_col].iloc[i:i+length]) if volume_col in data.columns else [None] * length
        }
        result.append(sliced_data)

# 将结果转换为 DataFrame
sliced_df = pd.DataFrame(result)

# 显示结果
print(sliced_df)