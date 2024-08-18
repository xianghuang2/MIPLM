import requests
import pandas as pd
from tqdm import tqdm
import time
import json

def google_search(query):
    api_key = 'API_KEY'
    cse_id = 'CSE_ID'
    url = 'https://www.googleapis.com/customsearch/v1'
    params = {
        'key': api_key,
        'cx': cse_id,
        'q': query,
        'num': 3  # 请求返回的结果数量
    }

    response = requests.get(url, params=params)
    results = response.json()
    return results


def format_row(row, miss_column):
    queries = []
    for col in row.index:
        query = "What is the " + miss_column + " of "
        if col != miss_column and col != "index":
            query += col + " is " + str(row[col]) + "?" 
            queries.append(query)
    return queries


def format_corrlation_buy(rag_res, row):
    rag_corr_prompt = '''
你是一个数据处理工程师，现在有一个缺失某个属性值的元组，缺失的数据用【M】表示，以及收集到的可能和填充这个属性值相关的文本，我将给你元组的观测信息，你需要判断这个文本和观测信息的相关性，并返回一个1-5之间的整数，表示相关性程度，数字越高则越相关。
例1：
元组信息：
属性名为：name,description,manufacturer,price
属性值为：Linksys EtherFast EZXS55W Ethernet Switch,5 x 10/100Base-TX LAN,【M】,无参考
相关文本：Cisco-Linksys EZXS55W EtherFast 10/100 5-Port
你的输出：3

例2：
元组信息：
属性名为：name,description,manufacturer,price
属性值为：Linksys EtherFast EZXS55W Ethernet Switch,5 x 10/100Base-TX LAN,【M】,无参考
相关文本：EZXS55W Datasheet
你的输出：1

我给你的数据信息为：
元组信息：
属性名为：name,description,manufacturer,price
属性值为：{},{},【M】,{}
相关文本：{}
'''.format(row['name'], row['description'], row['price'] if str(row['price']) != 'nan' else '无参考', 
           rag_res)
    return rag_corr_prompt
    

# def format_json(json_data, miss_column
    
# )

def excel_to_json(data_path, miss_column):
    df = pd.read_excel(data_path)
    json_data = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        queries = format_row(row, miss_column)
        all_res = []
        for query in queries:
            results = google_search(query)
            if not results:
                continue
            for idx, result in enumerate(results, 1):
                all_res.append(result['title'])
                json_data.append({
                    'index': row['index'],
                    'label': row[miss_column],
                    'row': row,
                    'rag_res': result['title']
                })
            with open('Buy.json', 'a', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
    return None

# excel_to_json("E:\code_test\Buy.xlsx", "manufacturer")



# 使用示例
query = "What is the manufacturer of name is Linksys EtherFast EZXS88W Ethernet Switch - EZXS88W?"
s_time = time.time()
# query_2 = "Who is the manufacturer described as Netgear ProSafe 16 Port 10/100 Rackmount Switch- JFS516NA?"
# query_3 = "What is the manufacturer of LaCie Pocket Floppy Disk Drive - 706018 and described as LaCie Pocket USB Floppy 1.44 MB?"
results = google_search(query)
e_time = time.time()
print("花费时间:", e_time-s_time)
# for idx, result in enumerate(results, 1):
#     print(f"{idx}. {result['title']}")
#     print(f"URL: {result['link']}\n")
