import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power
from data_processing.data_load import load_data


class LapRLSModel:
    def __init__(self, lb_s1, lb_s2, lb_s3, st_s1, st_s2, st_s3, A, pair_index):
        self.lb_s1 = lb_s1
        self.st_s1 = st_s1
        self.lb_s2 = lb_s2
        self.st_s2 = st_s2
        self.lb_s3 = lb_s3
        self.st_s3 = st_s3
        self.A = A
        self.pair_index = pair_index

    def linear_fusion(self):
        print(self.lb_s1.shape)
        print(self.lb_s2.shape)
        print(self.st_s1.shape)
        print(self.st_s2.shape)
        print("A shape:", self.A.shape)

        self.S_lb = 0.4 * self.lb_s1 + 0.4 * self.lb_s2 + 0.2 * self.lb_s3
        self.S_st = 0.4 * self.st_s1 + 0.4 * self.st_s2 + 0.2 * self.st_s3

    def normalize_similarity_matrix(self):
        S_lb_row_sum = np.sum(self.S_lb, axis=1)
        S_st_row_sum = np.sum(self.S_st, axis=1)
        D_lb = np.diag(S_lb_row_sum)
        D_st = np.diag(S_st_row_sum)


        self.S_nor_lb = fractional_matrix_power(D_lb, -0.5) @ self.S_lb @ fractional_matrix_power(D_lb, -0.5)
        self.S_nor_st = fractional_matrix_power(D_st, -0.5) @ self.S_st @ fractional_matrix_power(D_st, -0.5)

    def compute_interaction_predictor(self, p1=0.1, p2=0.1):
        S_lb_row_sum2 = np.sum(self.S_nor_lb, axis=1)
        S_st_row_sum2 = np.sum(self.S_nor_st, axis=1)
        D_lb2 = np.diag(S_lb_row_sum2)
        D_st2 = np.diag(S_st_row_sum2)

        L_lb = fractional_matrix_power(D_lb2, -0.5) @ (D_lb2 - self.S_nor_lb) @ fractional_matrix_power(D_lb2, -0.5)
        L_st = fractional_matrix_power(D_st2, -0.5) @ (D_st2 - self.S_nor_st) @ fractional_matrix_power(D_st2, -0.5)

        F_lb = self.S_nor_lb @ fractional_matrix_power(self.S_nor_lb + p1 * L_lb @ self.S_nor_lb, -1) @ self.A
        F_st = self.S_nor_st @ fractional_matrix_power(self.S_nor_st + p2 * L_st @ self.S_nor_st, -1) @ self.A.T
        self.F = ((F_lb + F_st.T) / 2).flatten()

    def save_results(self, output_path='LapRLS_pre.csv'):
        F_df = pd.DataFrame(self.F, index=self.pair_index, columns=["pre"])
        F_df.index.name = "index"  # 设置索引名字为 test_index
        F_df.to_csv(output_path, index=True, float_format='%.8f')

    def run(self):
        self.linear_fusion()
        self.normalize_similarity_matrix()
        self.compute_interaction_predictor()
        self.save_results()



lb_s1, lb_s2, lb_s3, st_s1, st_s2, st_s3, A, pair_index = load_data('./dataset/similarity_matrix.xlsx')


model = LapRLSModel(lb_s1, lb_s2, lb_s3, st_s1, st_s2, st_s3, A, pair_index)
model.run()