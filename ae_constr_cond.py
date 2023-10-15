import tensorflow as tf
import pandas as pd
import numpy as np



if __name__ == "__main__":
    cond_ji = np.zeros([267,267])
    rel_ij = np.zeros([267,267]) 
    df = pd.read_csv("./data_y_pred_true.csv",low_memory=False,header=None)
#     tmp = df.iloc[0]
#     print(list(tmp))
    total_rows = df.shape[0]
    print("------------total rows=",total_rows)
    for i in range(total_rows):
        cur_row = list(df.iloc[i])
        if(cur_row[267] == 0.0):
            continue
        else:
            cur_skill = int(cur_row[268])
            for j in range(267):
                cond_ji[cur_skill][j] = cur_row[j]
            cond_sum = np.sum(cond_ji,axis=0)
            for ind_i in range(267):
                for ind_j in range(267):
                    rel_ij[ind_i][ind_j] = cond_ji[ind_i][ind_j]/cond_sum[ind_j]
#                     if(rel_ij[ind_i][ind_j] != 0.0):
#                         print("ind_i=",ind_i,"ind_j=",ind_j,"rel_ij=",rel_ij[ind_i][ind_j])
        print("the row number=",i)
    for i in range(267):
        pd_rel = pd.DataFrame([rel_ij[i]])
        pd_rel.to_csv('./data_rel.csv', mode='a', header=False, index=None)
            
    



