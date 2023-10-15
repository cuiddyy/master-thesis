import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import socket

import gym

def num_to_one_hot(num, dim):
        # 将题目转换成one-hot的形式， 其中dim=num_skills * 2，前半段表示错误，后半段表示正确
        base = np.zeros(dim)
        if num >= 0:
            base[num] += 1
        return base


if __name__ == "__main__":
#     list1 = np.array([np.concatenate((num_to_one_hot(i,6),num_to_one_hot(i,6)))for i in [1,2,3,4,5]])
#     print("list1=",list1)
#     print("list2=",list2)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> correct_rate >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#     print("start!")
#     answer_sum = {}
#     correct_sum = {}
#     correct_rate = {}
#     with open("../data/ASSISTments_skill_builder_data.csv", 'r') as f:
#         for line in f:
#             fields = line.strip().split(",")  # 一个列表，[学生id，知识点id，答题结果]
#             if(fields[2] == "user_id"):
#                     continue
#             if(fields[16] == '' or fields[6] == ''):
#                 skill_id = 0
#                 answer = 1
#             else:
#                 skill_id = int(fields[16])
#                 answer = int(fields[6])
            
#             answer_sum[skill_id] = answer_sum.get(skill_id,0)+1
#             correct_sum[skill_id] = correct_sum.get(skill_id,0)+answer
#     for item in answer_sum.items():
#         correct_rate[item[0]] = correct_sum[item[0]]/item[1]
#     print("skill_len = ",len(answer_sum))
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1)
#     ax.scatter(correct_rate.keys(), correct_rate.values())
#     plt.xlabel("skill_id")
#     plt.ylabel("correct_rate")
#     plt.savefig('pic4.jpg')
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> cost_time_aver >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    
#     print("start!")
#     answer_cnt = {}
#     cost_time_sum = {}
#     cost_time_aver = {}
#     with open("../data/ASSISTments_skill_builder_data.csv", 'r') as f:
#         for line in f:
#             fields = line.strip().split(",")  # 一个列表，[学生id，知识点id，答题结果]
#             cost_time = 0
#             if(fields[2] == "user_id"):
#                     continue
#             if(fields[16] == '' or fields[22] == ''):
#                 skill_id = 0
#                 cost_time = 5
#             else:
#                 skill_id = int(fields[16])
#                 cost_time = int(fields[22])
#             answer_cnt[skill_id] = answer_cnt.get(skill_id,0)+1
#             cost_time_sum[skill_id] = cost_time_sum.get(skill_id,0)+(cost_time/1000)
#     for item in answer_cnt.items():
#         cost_time_aver[item[0]] = cost_time_sum[item[0]]/item[1]
#     print("skill_len = ",len(answer_cnt))
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1)
#     ax.scatter(cost_time_aver.keys(), cost_time_aver.values())
#     plt.xlabel("skill_id")
#     plt.ylabel("cost_time_aver/sec")
#     plt.savefig('cost_time_aver.jpg') 
    
#     # 计算综合难度
#     sum_diff = []
#     skill_ids = []
#     for item in cost_time_aver.items():
#         skill_ids.append(item[0])
#         sum_diff.append(0.4 * correct_rate[item[0]] + 0.6 * (item[1]/100))
#     pd_rel = pd.DataFrame({'skill_ids':skill_ids,'sum_diff':sum_diff})
#     pd_rel.to_csv('./sum_diff.csv', mode='a', index=None)    
    
    
#     df = pd.read_csv("../src/data_y_pred_true_skill.csv",low_memory=False)
#     sum_diff = df['sum_diff']
#     print("sum_diff = ",sum_diff)

    
    
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>socket通信>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    
#     x = (np.ones((5, 5)) * -1).astype(np.int32)
#     print(x)
#     output_graph_def = tf.GraphDef()
#     with open("./sevenSkillModel/saved_model.pb","rb") as f:
#         output_graph_def.ParseFromString(f.read())
#         _ = tf.import_graph_def(output_graph_def, name="")

#         node_in = sess.graph.get_tensor_by_name("input_node_name")
#         model_out = sess.graph.get_tensor_by_name("out_node_name")

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#     try:
#         sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM);
#         print("create socket succ!")

#         sock.bind(('localhost',8019))
#         print('bind socket succ!')

#         sock.listen(5)
#         print('listen succ!')

#     except:
#         print("init socket error!")

#     with tf.compat.v1.Session() as sess:
#         # load model
#         model_path= './saved_models/sevenSkillModel'
#         meta_graph_def = tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], model_path)

#         # get signature
#         signature = meta_graph_def.signature_def

#         # get tensor name
#         in_tensor_name = signature['predict'].inputs['input_x'].name
#         tar_id_tensor_name = signature['predict'].inputs['target_id'].name
#         kp_tensor_name = signature['predict'].inputs['keep_prob'].name
#         max_steps_tensor_name = signature['predict'].inputs['max_steps'].name
#         seq_len_tensor_name = signature['predict'].inputs['sequence_len'].name
#         out_tensor_name = signature['predict'].outputs['pred_all'].name

#         # get tensor
#         in_tensor = sess.graph.get_tensor_by_name(in_tensor_name)
#         tar_id_tensor = sess.graph.get_tensor_by_name(tar_id_tensor_name)
#         kp_tensor = sess.graph.get_tensor_by_name(kp_tensor_name)
#         max_steps_tensor = sess.graph.get_tensor_by_name(max_steps_tensor_name)
#         seq_len_tensor = sess.graph.get_tensor_by_name(seq_len_tensor_name)
#         out_tensor = sess.graph.get_tensor_by_name(out_tensor_name)
        
#         while(True):
#             print("listen for client...")
#             conn,addr=sock.accept()
#             print("get client")
#             print(addr)

#             conn.settimeout(30)
#             szBuf=conn.recv(1024)
#             print("recv:"+str(szBuf,'gbk'))

#             if "0"==szBuf:
#                 conn.send(b"exit")
#             else:
#                 # run
#                 input_tmp = np.array([[np.concatenate((num_to_one_hot(124+1,124*2),num_to_one_hot(1,10*2),num_to_one_hot(101+1,101*2)))]])
#                 tar_id = np.array([[1]])
#                 keep_pro = 0.8
#                 max_steps = 1
#                 seq_len = np.array([1])
#         #         print("input_tmp = ",input_tmp)
#                 output_tmp = sess.run(out_tensor, feed_dict={in_tensor: input_tmp,tar_id_tensor: tar_id,kp_tensor:keep_pro, max_steps_tensor: max_steps,seq_len_tensor: seq_len})
                
#                 str_out = ",".join(output_tmp)
                
#                 print(bytes(str_out, encoding="gbk"))
#                 print("____________________")
                
#                 conn.send(b"welcome client")
#                 conn.close()
#                 print("end of servive")
            
            
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#     df_r = pd.read_csv('../data/ASSISTments_skill_builder_data.csv')
#     df_r.dropna(subset=['skill_name'], inplace=True)
#     nums = df_r['skill_name']
#     set_nums = list(set(nums))
    
#     skill_id_name = {}
#     with open('../data/ASSISTments_skill_builder_data.csv', 'r') as f:
#             for line in f:
#                 fields = line.strip().split(",")  
#                 if(fields[2] == "user_id"):
#                     continue
#                 if(fields[16] == "" or fields[17] == ""):
#                     continue
#                 skill_id, skill_name = int(fields[16]), fields[17]
#                 if(not skill_id_name.__contains__(skill_id)):
#                     skill_id_name[skill_id] = skill_name
#                     print("skill_id = ",skill_id,"skill_name = ",skill_name)
#     id_list = []
#     name_list = []
#     for k,v in skill_id_name.items():
#         id_list.append(k)
#         name_list.append(v)
#     pd_w = pd.DataFrame({'skill_id': id_list,'skill_name': name_list})
#     pd_w.to_csv('./skill_names.csv', index=None,mode='w',sep=',') 
    
    
#     pd_w = pd.DataFrame(set_nums)
#     pd_w.to_csv('./skill_names.csv', mode='a', index=None) 
#     print("skill_size = ",len(set_nums))
#     print("set_nums = ",set_nums)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 从文件中读取数据，返回读取出来的数据和知识点个数
        # 保存每个学生的做题信息 {学生id: [[知识点id，答题结果], [知识点id，答题结果], ...]}，用一个二元列表来表示一个学生的答题信息
    skills = []  # 统计知识点的数量，之后输入的向量长度就是两倍的知识点数量
    hints = [] # 统计提示数量，提示数量的向量长度=提示数量取值数×2
    cost_times = [] #统计耗费时间，耗费时间的向量长度 = 101×2
    students = []
    count = 0
    with open("../data/ASSISTments_skill_builder_data.csv", 'r') as f:
        for line in f:
            fields = line.strip().split(",")  # 一个列表，[学生id，知识点id，答题结果，提示数量,花费时间]
            if(fields[2] == "user_id"):
                continue
            if(fields[2] == "" or fields[16] == "" or fields[6] == "" or fields[20] == "" or fields[22] == ""):
                continue
            count += 1
            student, skill, is_correct, hint, cost_time = int(fields[2]), int(fields[16]), int(fields[6]), int(fields[20]), int(fields[22])
            skills.append(skill)  # skill实际上是用该题所属知识点来表示的
            hints.append(hint)
            cost_times.append(cost_time)
            students.append(student)
    print("reco_num = ",count,"stu_num = ",len(list(set(students))),"skill_num = ",len(list(set(skills))))
    
    
    
    
    