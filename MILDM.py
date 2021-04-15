"""
@author: Inki
@contact: inki.yinji@qq.com
@version: Created in 2020 0903 1853, last modified in 2021 0415
@note: Literature you can refer to:
        1. The paper site --> https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8242668&tag=1
        2. The blog site --> https://blog.csdn.net/weixin_44575152/article/details/108139662
        This algorithms only can deal the binary classification.
"""

import numpy as np
import pandas as pd
import os as os

from collections import Counter
from FunctionTool import load_file, print_progress_bar
from I2B import max_similarity, ave_similarity
from Prototype import MIL


class MILDM(MIL):
    """
        The origin class of MIL, and all given vector data's type must be numpy.array.
        :param
            para_m:
                The number of selected mapping instance, and its default setting is 10.
            para_type:
                The types of MilDm algorithm:
                    1. 'ag': using all the training bags to generate the discriminative instances pool.
                    2. 'pg': only uses the positive bags.
                    3. 'al': selecting the most discriminative instance from each bag.
                    4. 'pl': only select the most discriminative instance form positive bag.
                And its default setting is 'ag'.
            para_sim_dis:
                The type of similarity function for bag and instance:
                    1. 'max': using the maximum distance between instance in bag and the discriminative instance.
                    2. To be continued...
                And its default setting is 'max'.
            para_ins_dis:
                The type of distance / similarity function for two instances.
                    1. 'euclidean': the euclidean distance.
                    2. 'rbf': the RBF kernel.
                And its default setting is 'rbf'.
            para_gamma:
                The gamma for RBF function.
            para_k:
                The k for k-th cross validation.
        @attribute:
            m:
                The number of selected mapping instance.
            type:
                The type of MilDm algorithm.
            algorithm_type:
                The type of algorithms.
            sim_dis_type:
                The type of similarity function.
            ins_dis_type:
                The type of distance function for instances.
        @example:
            # >>> temp_file_name = '../Data/Benchmark
            # >>> mil = MilDM(temp_file_name, para_ty
            # >>> temp_data_iter = mil.get_discriminative_ins()
        @note:
            If you given a error type, similarity function type or instances distance function,
            its will be setting the default value.
        """

    def __init__(self, para_path, para_m=None, para_type='ag', para_sim_dis='max',
                 para_ins_dis='rbf', para_gamma=1, para_k=10):
        """
        The constructor.
        """
        super(MILDM, self).__init__(para_path, True)
        self.m = para_m
        self.type = para_type
        self.sim_dis = para_sim_dis
        self.ins_dis = para_ins_dis
        self.gamma = para_gamma
        self.k = para_k
        self.full_mapping = []
        self.save_mapping = ''
        self.tr_idx = {}
        self.te_idx = {}
        self.num_tr_ins = {}
        self.num_pair = {}
        self.positive_label = 0
        self.negative_label = 0
        self.dis_bag2bag_path = 'D:/Data/TempData/DisOrSimilarity/'
        self.algorithm_type = ['ag', 'pg', 'al', 'pl']
        self.sim_dis_type = ['max', 'ave']
        self.ins_dis_type = ['rbf', 'euclidean', 'rbf2']
        self.__check_para()
        self.__initialize_mildm()

    def __initialize_mildm(self):
        """
        The initialize for MilDm.
        """
        temp_path = 'D:/Data/TempData/Mapping/MilDm/' + self.data_name + '_' + self.sim_dis + '_' + self.ins_dis
        if self.ins_dis == 'rbf' or self.ins_dis == 'rbf2':
            self.save_mapping = temp_path + '_' + str(self.gamma)
        elif self.ins_dis == 'euclidean':
            self.save_mapping = temp_path
        self.save_mapping += '.csv'

        self.tr_idx, self.te_idx = self.get_index(self.k)
        for i in range(self.k):
            self.num_tr_ins[i] = np.sum(self.bags_size[self.tr_idx[i]])
        self.positive_label = np.max(self.bags_label)
        self.negative_label = np.min(self.bags_label)

    def __check_para(self):
        """
        Check some parameters.
        """
        if self.type not in self.algorithm_type:
            self.type = self.algorithm_type[0]
        if self.sim_dis not in self.sim_dis_type:
            self.sim_dis = self.sim_dis_type[0]
        if self.ins_dis not in self.ins_dis_type:
            self.ins_dis = self.ins_dis_type[0]

    def __full_mapping(self):
        """
        Mapping bags by using all instances.
        @Note:
            The size of data set instance space will greatly affect the running time.
        """

        self.full_mapping = np.zeros((self.num_bags, self.num_ins))
        if not os.path.exists(self.save_mapping) or os.path.getsize(self.save_mapping) == 0:
            print("Full mapping starting...")
            open(self.save_mapping, 'a').close()

            for i in range(self.num_bags):
                print_progress_bar(i, self.num_bags)
                for j in range(self.num_ins):
                    self.full_mapping[i, j] = self.get_bag_ins_sim(i, j)
            pd.DataFrame.to_csv(pd.DataFrame(self.full_mapping), self.save_mapping,
                                index=False, header=False, float_format='%.6f')
            print("Full mapping end...")
        else:
            temp_data = load_file(self.save_mapping)
            for i in range(self.num_bags):
                self.full_mapping[i] = [float(value) for value in temp_data[i].strip().split(',')]

    def get_bag_ins_sim(self, i, j):
        """
        Get the similarity or distance between a bag and an instance.
        :param
            i:
                The index of i-th bag.
            j:
                The index of j-th instance.
        :return
            The similarity or distance between a bag and an instance.
        """
        if self.sim_dis == 'max':
            return max_similarity(self.bags[i, 0][:, :self.dimensions], self.ins[j], self.ins_dis, self.gamma)
        elif self.sim_dis == 'ave':
            return ave_similarity(self.bags[i, 0][:, :self.dimensions], self.ins[j], self.ins_dis, self.gamma)

    def refresh(self):
        """
        Refresh the initialize.
        """
        self.__initialize_mildm()

    def get_discriminative_ins(self):
        """
        Get the discriminative instance by compute score for each instance corresponding the type of MilDm.
        """

        # k-th cross validation.
        self.__full_mapping()
        for loop in range(self.k):
            # Step 1. Get the number of bag mapping must / cannot-link pairwise.
            temp_tr_idx = self.tr_idx[loop]
            temp_count = Counter(self.bags_label[temp_tr_idx])
            self.num_pair[loop] = [-1 / (temp_count[self.positive_label] ** 2 + temp_count[self.negative_label] ** 2),
                                   1 / (temp_count[self.positive_label] * temp_count[self.negative_label] * 2)]

            # Step 2. Get the label embedding matrix q.
            temp_num_tr_bag = len(self.tr_idx[loop])
            temp_q = np.zeros((temp_num_tr_bag, temp_num_tr_bag))
            for i in range(temp_num_tr_bag):
                for j in range(temp_num_tr_bag):
                    if self.bags_label[temp_tr_idx[i]] == self.bags_label[temp_tr_idx[j]]:
                        temp_q[i, j] = self.num_pair[loop][0]
                    else:
                        temp_q[i, j] = self.num_pair[loop][1]

            # Step 3. Get the row sum vector from q, i.e., d, and generate the diag matrix by using d.
            temp_d = np.diag(np.sum(temp_q, 1))

            # Step 4. Get the Laplacian matrix by using (d - q).
            temp_l = temp_d - temp_q
            del temp_d, temp_q

            # Step 5. Compute score of the instances corresponding the algorithm type.
            temp_mapping = np.zeros((temp_num_tr_bag, self.num_tr_ins[loop]))
            temp_ins_lab = np.zeros(self.num_tr_ins[loop], dtype=int)
            temp_ins_sta = np.zeros_like(temp_ins_lab)

            temp_idx = 0
            for i in range(temp_num_tr_bag):
                temp_list = list(range(self.ins_idx[temp_tr_idx[i]],
                                       self.ins_idx[temp_tr_idx[i]] + self.bags_size[temp_tr_idx[i]]))
                temp_mapping[:, temp_idx: temp_idx + self.bags_size[temp_tr_idx[i]]] = \
                    self.full_mapping[temp_tr_idx, :][:, temp_list]
                temp_ins_lab[temp_idx: temp_idx + self.bags_size[temp_tr_idx[i]]] = \
                    np.tile(temp_tr_idx[i], (1, self.bags_size[temp_tr_idx[i]]))
                temp_ins_sta[temp_idx: temp_idx + self.bags_size[temp_tr_idx[i]]] = \
                    np.arange(self.bags_size[temp_tr_idx[i]])
                temp_idx += self.bags_size[temp_tr_idx[i]]

            # Step 6. Get the score for each training instances.
            temp_score = np.zeros(self.num_tr_ins[loop])
            for i in range(self.num_tr_ins[loop]):
                temp_score[i] = np.dot(np.dot(temp_mapping[:, i], temp_l), temp_mapping[:, i])
            del temp_l

            temp_idx = 0
            temp_ins_idx = []
            temp_max_score_idx = np.argsort(temp_score)[::-1]
            if self.m is None:
                self.m = temp_num_tr_bag
            if self.type == 'ag':
                temp_max_score_idx = temp_max_score_idx[: self.m]
                temp_ins_idx = [[temp_ins_lab[temp_max_score_idx[i]], temp_ins_sta[temp_max_score_idx[i]]]
                                for i in range(self.m)]
            elif self.type == 'pg':
                for idx in temp_max_score_idx:
                    if self.bags_label[temp_ins_lab[idx]] == self.positive_label and len(temp_ins_idx) < self.m:
                        temp_ins_idx.append([temp_ins_lab[idx], temp_ins_sta[idx]])
            elif self.type == 'al':
                for i in range(temp_num_tr_bag):
                    temp_max_bag_idx = np.argmax(temp_score[temp_idx: temp_idx + self.bags_size[temp_tr_idx[i]]])
                    temp_idx += self.bags_size[temp_tr_idx[i]]
                    temp_ins_idx.append([temp_tr_idx[i], temp_max_bag_idx])
            elif self.type == 'pl':
                for i in range(temp_num_tr_bag):
                    temp_max_bag_idx = np.argmax(temp_score[temp_idx: temp_idx + self.bags_size[temp_tr_idx[i]]])
                    temp_idx += self.bags_size[temp_tr_idx[i]]
                    if self.bags_label[temp_tr_idx[i]] == self.positive_label:
                        temp_ins_idx.append([temp_tr_idx[i], temp_max_bag_idx])

            # Step 7. Mapping bags by using maximum score instances.
            temp_num_ins = len(temp_ins_idx)
            temp_num_te_bag = len(self.te_idx[loop])
            ret_tr_ins = np.zeros((temp_num_tr_bag, temp_num_ins))
            ret_tr_label = self.bags_label[self.tr_idx[loop]]
            ret_te_ins = np.zeros((temp_num_te_bag, temp_num_ins))
            ret_te_label = self.bags_label[self.te_idx[loop]]

            for j in range(temp_num_ins):
                temp_idx = self.ins_idx[temp_ins_idx[j][0]] + temp_ins_idx[j][1]
                for i in range(temp_num_tr_bag):
                    ret_tr_ins[i, j] = self.full_mapping[temp_tr_idx[i], temp_idx]
                for i in range(temp_num_te_bag):
                    ret_te_ins[i, j] = self.full_mapping[self.te_idx[loop][i], temp_idx]
            yield ret_tr_ins, ret_tr_label, ret_te_ins, ret_te_label, None
