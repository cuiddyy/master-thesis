import gym

class KnowledgeEnv(gym.Env):
    def __init__(self):
        self.a1 = 0.4
        self.a2 = -0.2
        self.a3 = 0.1
        self.stu_par = 0.3
        self.pre_skill_status = []
        self.max_consid = 5
        
        df = pd.read_csv("../src/sum_diff.csv",low_memory=False)
        self.sum_diff = df['sum_diff']
        
        self.conds = []
        with open("./data_rel.csv", 'r') as f:
            for line in f:
                tmp_list = []
                fields = line.strip().split(",")  # [各个知识点之间的答对条件概率]
                for flo_num in fields:                
                    tmp_list.append(float(flo_num))
                self.conds.append(tmp_list)
            
    def dkt_model(self,input_tmp):
        with tf.compat.v1.Session() as sess:
            # load model
            model_path= './saved_models/sevenSkillModel'
            meta_graph_def = tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], model_path)

            # get signature
            signature = meta_graph_def.signature_def

            # get tensor name
            in_tensor_name = signature['predict'].inputs['input_x'].name
            tar_id_tensor_name = signature['predict'].inputs['target_id'].name
            kp_tensor_name = signature['predict'].inputs['keep_prob'].name
            max_steps_tensor_name = signature['predict'].inputs['max_steps'].name
            seq_len_tensor_name = signature['predict'].inputs['sequence_len'].name
            out_tensor_name = signature['predict'].outputs['pred_all'].name

            # get tensor
            in_tensor = sess.graph.get_tensor_by_name(in_tensor_name)
            tar_id_tensor = sess.graph.get_tensor_by_name(tar_id_tensor_name)
            kp_tensor = sess.graph.get_tensor_by_name(kp_tensor_name)
            max_steps_tensor = sess.graph.get_tensor_by_name(max_steps_tensor_name)
            seq_len_tensor = sess.graph.get_tensor_by_name(seq_len_tensor_name)
            out_tensor = sess.graph.get_tensor_by_name(out_tensor_name)

            # run
        #         input_tmp = np.array([[np.concatenate((num_to_one_hot(124+action,124*2),num_to_one_hot(10+2,10*2),num_to_one_hot(101+2,101*2)))]])
            tar_id = np.array([[1]])
            keep_pro = 0.8
            max_steps = 1
            seq_len = np.array([1])
        #         print("input_tmp = ",input_tmp)
            output_tmp = sess.run(out_tensor, feed_dict={in_tensor: input_tmp,tar_id_tensor: tar_id,kp_tensor:keep_pro, max_steps_tensor: max_steps,seq_len_tensor: seq_len})
            return output_tmp
#         print(len(output_tmp[0][0]))
    def reset(self):
        self.cur_status = []
        self.pre_skill = -1
        for i in range(SKILLS_NUM):
            self.cur_skill_status.append(0.5)
        return self.cur_skill_status
    def step(self,action):
        reward = 0.0
        done = False
        if(self.pre_skill == -1):
            r1 = 0
        else:
            r1 = self.a1 * conds(action,pre_skill)
        if(self.pre_skill == -1):
            r2 = 0
        else:
            r2 = self.a2 * (self.sum_diff(action)-self.sum_diff(self.pre_skill))^2
        tmp_r = 0.0
#         inputs = tf.ont_hot(SKILLS_NUM*2,action) #默认是答对了
        input_tmp = np.array([[np.concatenate((num_to_one_hot(124+action,124*2),num_to_one_hot(1,10*2),num_to_one_hot(101+2,101*2)))]])
        self.cur_status = self.dkt_model(input_tmp)
        pre_skill_status.append(cur_status[action])
        if(self.pre_skill_status.len <= self.max_consid):
            tmp_r = sum(self.pre_skill_status)/self.pre_skill_status.len
        else:
            pre_skill_status.pop(0)
            tmp_r = sum(self.pre_skill_status)/self.max_consid
        r3 = 1-abs(self.stu_par - tmp_r)
        r3 *= a3
        reward = r1 + r2 + r3
        pre_skill = action
        
        cnt = 0
        for stat in self.cur_status:
            if(stat > 0.5):
                cnt += 1
        if(cnt >= 74):
            done = True
        
        return self.cur_status, reward, done, {}
