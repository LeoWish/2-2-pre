import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 1. 读取Excel文件中的KEGG_lb工作表
file_path = "../dataset/similarity_matrix.xlsx"  # 替换为你的文件路径
kegg_data = pd.read_excel(file_path, sheet_name="KEGG_lb")

# 2. 提取数值数据（假设第一列是样本名或标签，其余为数值）
# 如果数据没有列名，可以跳过header参数
data_values = kegg_data.iloc[:, 1:].values  # 跳过第一列（假设是标签）

# 3. 计算余弦相似性
cosine_sim = cosine_similarity(data_values)

# 4. 将结果转换为DataFrame（添加行列标签）
sample_names = kegg_data.iloc[:, 0].tolist()  # 假设第一列是样本名
cosine_df = pd.DataFrame(
    cosine_sim,
    index=sample_names,
    columns=sample_names
)

# 5. 保存为新的Excel文件
output_path = "kegg_cosine_similarity_lb.xlsx"
cosine_df.to_excel(output_path, sheet_name="Cosine_Similarity")

print(f"余弦相似性矩阵已保存至: {output_path}")