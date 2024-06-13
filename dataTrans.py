# 将data中的txt文件转为csv文件，便于后续处理
import csv
order = input("请输入你要处理的文件：")
if order == 'train':
    input_file_path = 'D:/machine-learning/Recommend-System/data/train.txt'
    output_file_path = 'D:/machine-learning/Recommend-System/data/train.csv'

    # 打开输入文件
    with open(input_file_path, 'r') as infile:
        lines = infile.readlines()

    # 初始化输出列表
    output_data = []

    # 解析输入文件内容
    current_user_id = None
    for line in lines:
        if '|' in line:
            # 解析用户信息
            current_user_id = line.split('|')[0]
        else:
            # 解析项目评分信息
            item_id, score = line.split()
            output_data.append([current_user_id, item_id, score])

    # 写入到CSV文件
    with open(output_file_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        # 写入表头
        writer.writerow(['user_id', 'item_id', 'score'])
        # 写入数据
        writer.writerows(output_data)

    print(f'转换完成，CSV文件保存在：{output_file_path}')


# test.txt 文件中没有score，其余一样
if order == 'test':
    input_file_path = 'D:/machine-learning/Recommend-System/data/test.txt'
    output_file_path = 'D:/machine-learning/Recommend-System/data/test.csv'

    with open(input_file_path, 'r') as infile:
        lines = infile.readlines()

    output_data = []

    current_user_id = None
    for line in lines:
        if '|' in line:
            # 解析用户信息
            current_user_id = line.split('|')[0]
        else:
            # 只有id,score待预测
            item_id = int(line.strip())
            output_data.append([current_user_id, item_id])

    with open(output_file_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['user_id', 'item_id', 'score'])
        writer.writerows(output_data)

    print(f'转换完成，CSV文件保存在：{output_file_path}')

if order == 'itemAttribute':
    input_file_path = 'D:/machine-learning/Recommend-System/data/itemAttribute.txt'
    output_file_path = 'D:/machine-learning/Recommend-System/data/itemAttribute.csv'

    with open(input_file_path, 'r') as infile:
        lines = infile.readlines()

    # 初始化输出列表
    output_data = []

    # 解析输入文件内容
    for line in lines:
        item_id, attribute_1, attribute_2 = line.strip().split('|')
        
        # 将 'None' 替换为 '0'
        if attribute_1 == 'None':
            attribute_1 = '0'
        if attribute_2 == 'None':
            attribute_2 = '0'
        
        output_data.append([item_id, attribute_1, attribute_2])

    with open(output_file_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['item_id', 'attribute_1', 'attribute_2'])
        writer.writerows(output_data)

    print(f'转换完成，CSV文件保存在：{output_file_path}')