import torch
from torch.utils.data import Dataset
import numpy as np


class TextDataset(Dataset):
    def __init__(self, users_list, labels_list, words_id, reddit_data_list, max_posts=100, max_words_per_post=200):
        """
        Args:
            users_list: List[str], 用户名列表（用于查找 reddit_data）
            labels_list: List[int], 对应标签
            words_id: Dict[word -> idx], 词汇表
            reddit_data_list: 原始 pkl 加载的数据列表，每个元素是 {'user': ..., 'subreddit': [...], 'label': ...}
            max_posts: int, 每个用户最多保留多少篇帖子
            max_words_per_post: int, 每篇帖子最多保留多少个词
        """
        self.words_id = words_id
        self.max_posts = max_posts
        self.max_words_per_post = max_words_per_post
        self.PAD_ID = words_id["<PAD>"]
        self.UNK_ID = words_id["<UNK>"]

        # 构建 user -> data 映射
        user_to_data = {item['user']: item for item in reddit_data_list}

        self.samples = []
        for user, label in zip(users_list, labels_list):
            if user not in user_to_data:
                continue
            raw_data = user_to_data[user]
            posts = raw_data['subreddit']  # list of strings

            # 存储 tokenized 后的帖子序列
            tokenized_posts = []

            for post in posts:
                words = post.strip().split()[:max_words_per_post]  # 截断长度
                word_ids = [
                    words_id.get(w, self.UNK_ID)
                    for w in words
                ]
                # 补齐到 max_words_per_post
                if len(word_ids) < max_words_per_post:
                    word_ids += [self.PAD_ID] * (max_words_per_post - len(word_ids))
                tokenized_posts.append(word_ids)

            # 只保留前 max_posts 篇
            if len(tokenized_posts) > max_posts:
                tokenized_posts = tokenized_posts[:max_posts]
            else:
                # 补齐帖子数量
                while len(tokenized_posts) < max_posts:
                    tokenized_posts.append([self.PAD_ID] * max_words_per_post)

            # 转为 numpy 数组
            post_matrix = np.array(tokenized_posts, dtype=np.int64)  # (T, L)

            # 构造 mask: 帖子级有效 mask (T,), 词级 mask (T, L)
            text_mask = [int(not all(w == self.PAD_ID for w in post)) for post in tokenized_posts]  # T,
            post_mask = [[int(w != self.PAD_ID) for w in post] for post in tokenized_posts]  # (T, L)

            self.samples.append({
                'input_ids': post_matrix,         # (T, L)
                'text_mask': np.array(text_mask), # (T,)
                'post_mask': np.array(post_mask), # (T, L)
                'label': label
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
# #本地运行
# def collate_fn(batch):
#     """自定义批处理函数"""
#     batch_size = len(batch)
#     max_posts = batch[0]['input_ids'].shape[0]   # T
#     max_words = batch[0]['input_ids'].shape[1]   # L
#
#     # 初始化 batch 张量
#     input_ids = torch.zeros((batch_size, max_posts, max_words), dtype=torch.long)
#     text_masks = torch.zeros((batch_size, max_posts), dtype=torch.float)
#     post_masks = torch.zeros((batch_size, max_posts, max_words), dtype=torch.float)
#     labels = torch.zeros(batch_size, dtype=torch.long)
#
#     for i, sample in enumerate(batch):
#         input_ids[i] = torch.from_numpy(sample['input_ids'])
#         text_masks[i] = torch.from_numpy(sample['text_mask']).float()      # ← 加 .float()
#         post_masks[i] = torch.from_numpy(sample['post_mask']).float()      # ← 加 .float()
#         labels[i] = sample['label']
#     return input_ids, labels, text_masks, post_masks
#服务器运行
def collate_fn(batch):
    """自定义批处理函数（修复数据类型问题版）

    参数:
        batch: 包含多个样本的列表，每个样本是字典格式

    返回:
        tuple: (input_ids, labels, text_masks, post_masks)
    """
    if not batch:
        raise ValueError("Batch cannot be empty")

    # 获取batch维度和最大长度
    batch_size = len(batch)
    example = batch[0]

    # 动态获取最大长度
    max_posts = example['input_ids'].shape[0]  # T (帖子数)
    max_words = example['input_ids'].shape[1]  # L (单词数)

    # 初始化batch张量（明确指定所有dtype）
    input_ids = torch.zeros((batch_size, max_posts, max_words), dtype=torch.int64)
    text_masks = torch.zeros((batch_size, max_posts), dtype=torch.float32)
    post_masks = torch.zeros((batch_size, max_posts, max_words), dtype=torch.float32)
    labels = torch.zeros(batch_size, dtype=torch.int64)

    for i, sample in enumerate(batch):
        # 明确指定所有转换的数据类型
        input_ids[i] = torch.tensor(sample['input_ids'], dtype=torch.int64)
        text_masks[i] = torch.tensor(sample['text_mask'], dtype=torch.float32)
        post_masks[i] = torch.tensor(sample['post_mask'], dtype=torch.float32)
        labels[i] = torch.tensor(sample['label'], dtype=torch.int64)

    return input_ids, labels, text_masks, post_masks