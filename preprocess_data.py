import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import random
from loguru import logger


def split_data(input_file, train_file, dev_file, test_file):
    """
    将训练数据，切分为train/dev/test三份数据
    """
    dev_size = 500
    test_size = 1000
    df = pd.read_csv(input_file, sep=',')
    logger.info("len of input data:{}".format(len(df)))
    label2rows = defaultdict(list)
    rows = df.to_dict('records')

    # 收集每个label_group下的数据集合
    for row in tqdm(rows):
        label = row['label_group']
        label2rows[label].append(row)
    label2rows = list(label2rows.items())
    random.shuffle(label2rows)

    # 保存切分后的数据
    dev_rows = []
    for label, rows in label2rows[:dev_size]:
        dev_rows += rows
    df_dev = pd.DataFrame(dev_rows)
    df_dev.to_csv(dev_file)

    test_rows = []
    for label, rows in label2rows[dev_size: dev_size + test_size]:
        test_rows += rows
    df_test = pd.DataFrame(test_rows)
    df_test.to_csv(test_file)

    train_rows = []
    for label, rows in label2rows[dev_size + test_size:]:
        train_rows += rows
    df_train = pd.DataFrame(train_rows)
    df_train.to_csv(train_file)
    print('dev len:{}'.format(len(dev_rows)))
    print('test len:{}'.format(len(test_rows)))
    print('train len:{}'.format(len(train_rows)))


if __name__ == '__main__':
    pass
