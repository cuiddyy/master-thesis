"""
1 在实例化该类之后，要调用gen_attr函数来生成未处理的数据集
2 如果需要生成batch_size的数据，则可以调用next_batch方法，如果想把所有数据直接导入，则可以用format_data方法来输出
"""
#encoding: utf-8

import numpy as np
import random


class DataGenerator(object):
    # 导入的seqs是train_seqs，或者是test_seqs
    def __init__(self, fileName, config):
        self.fileName = fileName
        self.train_seqs = []
        self.test_seqs = []
        self.infer_seqs = []
        self.batch_size = config.batch_size
        self.pos = 0
        self.end = False
        self.num_skills = config.num_skills
        self.skills_to_int = {}  # 知识点到索引的映射
        self.int_to_skills = {}  # 索引到知识点的映射
        self.hint_max = config.hint_max
        self.cost_time_max = config.cost_time_max

    def read_file(self):
        # 从文件中读取数据，返回读取出来的数据和知识点个数
        # 保存每个学生的做题信息 {学生id: [[知识点id，答题结果], [知识点id，答题结果], ...]}，用一个二元列表来表示一个学生的答题信息
        seqs_by_student = {}
        skills = []  # 统计知识点的数量，之后输入的向量长度就是两倍的知识点数量
        hints = [] # 统计提示数量，提示数量的向量长度=提示数量取值数×2
        cost_times = [] #统计耗费时间，耗费时间的向量长度 = 101×2
        with open(self.fileName, 'r') as f:
            for line in f:
                fields = line.strip().split(",")  # 一个列表，[学生id，知识点id，答题结果，提示数量,花费时间]
                if(fields[2] == "user_id"):
                    continue
                if(fields[2] == "" or fields[16] == "" or fields[6] == "" or fields[20] == "" or fields[22] == ""):
                    continue
                student, skill, is_correct, hint, cost_time = int(fields[2]), int(fields[16]), int(fields[6]), int(fields[20]), int(fields[22])
                skills.append(skill)  # skill实际上是用该题所属知识点来表示的
                hints.append(hint)

                cost_times.append(cost_time)
                seqs_by_student[student] = seqs_by_student.get(student, []) + [[skill, is_correct, hint, cost_time]]  # 保存每个学生的做题信息
#         for i in seqs_by_student:
#             if(len(seqs_by_student[i]) == 1):
#                 print("i == ",i,"seqs_by_student[i] == ",seqs_by_student[i])
        return seqs_by_student, list(set(skills)), list(set(hints)), list(set(cost_times))

    def gen_dict(self, unique_skills):
        """
        构建知识点映射表，将知识点id映射到[0, 1, 2...]表示
        :param unique_skills: 无重复的知识点列表
        :return:
        """
        sorted_skills = sorted(unique_skills)
        skills_to_int = {}
        int_to_skills = {}
        for i in range(len(sorted_skills)):
            skills_to_int[sorted_skills[i]] = i
            int_to_skills[i] = sorted_skills[i]

        self.skills_to_int = skills_to_int
        self.int_to_skills = int_to_skills
    def gen_hint_dict(self,unique_hints):
        sorted_hints = sorted(unique_hints)
        hints_to_int = {}
        int_to_hints = {}
        for i in range(len(sorted_hints)):
            hints_to_int[sorted_hints[i]] = i
            int_to_hints[i] = sorted_hints[i]
        self.hints_to_int = hints_to_int
        self.int_to_hints = int_to_hints
    def gen_time_dict(self,unique_times):
        sorted_times = sorted(unique_times)
        times_to_int = {}
        int_to_times = {}
        for i in sorted_times:
            # 耗费时间映射
            ind = 0
            if(i < 0):
                ind = 0
            else if(i >= 3000000):
                ind = 100
            else:
                ind = i/1000/30
            times_to_int[i] = ind
            int_to_times[ind] = i
    
    def split_dataset(self, seqs_by_student, sample_rate=0.2, random_seed=1):
        # 将数据分割成测试集和训练集
        sorted_keys = sorted(seqs_by_student.keys())  # 得到排好序的学生id的列表

        random.seed(random_seed)
        # 随机抽取学生id，将这部分学生作为测试集
        test_keys = set(random.sample(sorted_keys, int(len(sorted_keys) * sample_rate)))

        # 此时是一个三层的列表来表示的，最外层的列表中的每一个列表表示一个学生的做题信息
        test_seqs = [seqs_by_student[k] for k in seqs_by_student if k in test_keys]
        train_seqs = [seqs_by_student[k] for k in seqs_by_student if k not in test_keys]
        return train_seqs, test_seqs

    def gen_attr(self, is_infer=False):
        """
        生成待处理的数据集
        :param is_infer: 判断当前是训练模型还是利用模型进行预测
        :return:
        """
        if is_infer:
            seqs_by_students, skills, hints, cost_times = self.read_file()
            infer_seqs = [value for value in seqs_by_students.values()]
            self.infer_seqs = infer_seqs
        else:
            seqs_by_students, skills, hints, cost_times = self.read_file()
            train_seqs, test_seqs = self.split_dataset(seqs_by_students)
            self.train_seqs = train_seqs
            self.test_seqs = test_seqs

        self.gen_dict(skills)  # 生成知识点到索引的映射字典
        self.gen_hint_dict(hints) # 生成提示数量到索引的映射字典
        self.gen_time_dict(cost_times) # 生成耗费时间到索引的映射字典

    def pad_sequences(self, sequences, maxlen=None, value=0.):
        # 按每个batch中最长的序列进行补全, 传入的sequences是二层列表
        # 统计一个batch中每个序列的长度，其实等于seqs_len
        lengths = [len(s) for s in sequences]
        # 统计下该batch中序列的数量
        nb_samples = len(sequences)
        # 如果没有传入maxlen参数就自动获取最大的序列长度
        if maxlen is None:
            maxlen = np.max(lengths)
        # 构建x矩阵
        x = (np.ones((nb_samples, maxlen)) * value).astype(np.int32)

        # 遍历batch，去除每一个序列
        for idx, s in enumerate(sequences):
            trunc = np.asarray(s, dtype=np.int32)
            x[idx, :len(trunc)] = trunc

        return x
    def pad_sequences_2d(self, sequences, maxlen=None, value=0.):
        #按每个batch中最长的序列进行补全, 传入的sequences是二层列表
        # 统计一个batch中每个序列的长度，其实等于seqs_len
        lengths = [len(s) for s in sequences]
        # 统计下该batch中序列的数量
        nb_samples = len(sequences)
        # 如果没有传入maxlen参数就自动获取最大的序列长度
        if maxlen is None:
            maxlen = np.max(lengths)
        # 构建x矩阵
        x = (np.ones((nb_samples, maxlen, 2)) * value).astype(np.int32)
#         print("pad_sequences_2d_sequences=",sequences)
        # 遍历batch，去除每一个序列
        for idx, s in enumerate(sequences):
#             print("s=",s)
            trunc = np.asarray(s, dtype=np.int32)
            if(len(trunc) == 0):
                continue
#             print("x[idx, :len(trunc)]=",x[idx, :len(trunc)],"trunc=",trunc)
            x[idx, :len(trunc)] = trunc
        return x
    def num_to_one_hot(self, num, dim):
        # 将题目转换成one-hot的形式， 其中dim=num_skills * 2，前半段表示错误，后半段表示正确
        base = np.zeros(dim)
        if num >= 0:
            base[num] += 1
        return base

    def format_data(self, seqs):
        # 生成输入数据和输出数据，输入数据是每条序列的前n-1个元素，输出数据是每条序列的后n-1个元素
#         for seq in seqs:
#             if(len(seq) == 0):
#                 print("seqs[i] == ",seq)
#         print("seqs=",len(seqs))
        # 统计一个batch_size中每条序列的长度，在这里不对序列固定长度，通过条用tf.nn.dynamic_rnn让序列长度可以不固定
        seq_len = np.array(list(map(lambda seq: len(seq) - 1, seqs)))
#         print("seqs=",seqs)
        max_len = max(seq_len)  # 获得一个batch_size中最大的长度
        # i表示第i条数据，j只从0到len(i)-1，x作为输入只取前len(i)-1个，sequences=[j[0] + num_skills * j[1], ....]
        # 此时要将知识点id j[0] 转换成index表示
        x_sequences = np.array([[[self.skills_to_int[j[0]] + self.num_skills * j[1],self.hints_to_int[j[2]] + self.hint_max * j[1],self.times_to_int[j[3]] + self.cost_time_max * j[1]] for j in i[:-1]] for i in seqs])
#         for i in seqs:
#             if(len(i) == 1):
#                 print("first>>>>>>>>>>>>>>>>>>>>>>>>>>>>>i = ",i)
#             for j in i[:-1]:
                
#                 tmp_list = [self.skills_to_int[j[0]] + self.num_skills * j[1],self.hints_to_int[j[2]] + self.hint_max * j[1]]
# #                 if(tmp_list == []):
#                 print("tmp_list>>>>>>>>>>>>>>>>>>>>>",tmp_list)
#             if(len(i) == 1):
#                 print("second>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
#         for idx,seq in enumerate(x_sequences):
#             if(len(seq) == 0):
#                 print("seq = ",seq)
#                 break
#         print("x_sequences=",len(x_sequences))
#         print("x_sequences=",x_sequences)
        # 将输入的序列用-1进行补全，补全后的长度为当前batch的最大序列长度
        x = self.pad_sequences_2d(x_sequences, maxlen=max_len, value=-1)

        # 构建输入值input_x，x为一个二层列表，i表示一个学生的做题信息，也就是一个序列，j就是一道题的信息
        input_x = np.array([[np.concatenate((self.num_to_one_hot(j[0], self.num_skills * 2),self.num_to_one_hot(j[1], self.hint_max * 2),self.num_to_one_hot(j[2],self.cost_time_max * 2))) for j in i] for i in x])

        # 遍历batch_size，然后取每条序列的后len(i)-1 个元素中的知识点id为target_id
        source_id_seqs = np.array([[self.skills_to_int[j[0]] for j in i[:-1]] for i in seqs])
        source_id = self.pad_sequences(source_id_seqs, maxlen=max_len, value=0)
        # 同source_id
        source_correctness_seqs = np.array([[j[1] for j in i[:-1]] for i in seqs])
        source_correctness = self.pad_sequences(source_correctness_seqs, maxlen=max_len, value=0)

        # 遍历batch_size，然后取每条序列的后len(i)-1 个元素中的知识点id为target_id
        target_id_seqs = np.array([[self.skills_to_int[j[0]] for j in i[1:]] for i in seqs])
        target_id = self.pad_sequences(target_id_seqs, maxlen=max_len, value=0)

        # 同target_id
        target_correctness_seqs = np.array([[j[1] for j in i[1:]] for i in seqs])
        target_correctness = self.pad_sequences(target_correctness_seqs, maxlen=max_len, value=0)

        return dict(input_x=input_x, target_id=target_id, target_correctness=target_correctness, source_id=source_id,
                    source_correctness=source_correctness, seq_len=seq_len, max_len=max_len)

    def next_batch(self, seqs, mode):
        """
        采用生成器的形式生成一个batch
        :param seqs:
        :param mode: 判断是测试，训练还是验证
        :return:
        """

        length = len(seqs)
        num_batchs = length // self.batch_size
        if mode == "infer" or mode == "test":
            # 如果是测试或验证，则把所有的数据都用完，最后一个batch的大小若小于batch_size，也直接使用
            num_batchs += 1
        start = 0
        print(">>>>>>>>>>>>>>>>>>>>>>num_batchs=",num_batchs)
        for i in range(num_batchs):

            batch_seqs = seqs[start: start + self.batch_size]
            if(len(batch_seqs) == 0):
                continue
            start += self.batch_size
            params = self.format_data(batch_seqs)

            yield params

