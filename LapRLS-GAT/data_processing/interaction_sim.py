import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def interaction_sim():
    path = '../dataset/similarity_matrix.xlsx'
    sheet = 'A'
    file_path = path
    df = pd.read_excel(file_path, sheet_name=sheet, index_col=0)

    B_interaction_matrix = np.dot(df.T, df)

    A_interaction_matrix = np.dot(df, df.T)

    np.fill_diagonal(B_interaction_matrix, 1)
    np.fill_diagonal(A_interaction_matrix, 1)

    B_cosine_similarity = cosine_similarity(B_interaction_matrix)
    A_cosine_similarity = cosine_similarity(A_interaction_matrix)

    # 将结果保存为CSV文件
    B_interaction_matrix_df = pd.DataFrame(B_interaction_matrix, index=df.columns, columns=df.columns)
    A_interaction_matrix_df = pd.DataFrame(A_interaction_matrix, index=df.index, columns=df.index)

    B_cosine_similarity_df = pd.DataFrame(B_cosine_similarity, index=df.columns, columns=df.columns)
    A_cosine_similarity_df = pd.DataFrame(A_cosine_similarity, index=df.index, columns=df.index)

    # B_interaction_matrix_df.to_csv('../temp_data/B_interaction_matrix.csv', float_format='%.6f')
    # A_interaction_matrix_df.to_csv('../temp_data/A_interaction_matrix.csv', float_format='%.6f')
    B_cosine_similarity_df.to_csv('../target_cosine_similarity.csv', float_format='%.6f')
    A_cosine_similarity_df.to_csv('../drug_cosine_similarity.csv', float_format='%.6f')

    # print("B的互作关联矩阵已保存为B_interaction_matrix.csv")
    # print("A的互作关联矩阵已保存为A_interaction_matrix.csv")
    print("target的互作关联余弦相似性矩阵已保存为target_cosine_similarity.csv")
    print("drug的互作关联余弦相似性矩阵已保存为drug_cosine_similarity.csv")

interaction_sim()