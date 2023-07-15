import glob
import os
import re
import statistics as stat
from scipy.stats import ttest_ind

#inp = ["dsads", "pamap", "hapt",]
inp = ["Model_Report"]
mem = [10, 20, 40]
'''
def mem_utils():
    for inp_ in inp:
        for mem_ in mem:
            micro, macro = [], []
            for i in range(1, 6):
                path = f"gmm-max/{inp_}_{mem_}_{i}.txt"
                with open(path, 'r') as f:
                    for line in f.readlines():
                        if line.strip().startswith("f1_score(micro)"):
                            micro.append(float(line.strip().split(":")[1]))
                        if line.strip().startswith("f1_score(macro)"):
                            macro.append(float(line.strip().split(":")[1]))
            f1_micro_mean, f1_micro_std, f1_micro_var = stat.mean(micro), stat.pstdev(micro), stat.pvariance(micro)
            f1_macro_mean, f1_macro_std, f1_macro_var = stat.mean(macro), stat.pstdev(macro), stat.pvariance(macro)
            print(f"{inp_}_{mem_}\tf1_micro: {f1_micro_mean:.2f} ({f1_micro_std:.2f}/{f1_micro_var:.2f})\tf1_macro: {f1_macro_mean:.2f} ({f1_macro_std:.2f}/{f1_macro_var:.2f})")        
        print()    
'''
def non_mem_utils():
    for inp_ in inp:
        time1, micro1, macro1 = [], [], []
        for i in range(1, 11):
            path = f"Figures_A/60/Figures_M_3/T{i}/{inp_}.txt"
            with open(path, 'r') as f:
                for line in f.readlines():
                    if line.strip().startswith("Training"):
                        time1.append(float(line.strip().split("s")[1]))
                    if line.strip().startswith("A"):
                        micro1.append(0.01 * float(line.strip().split(".")[1]))
                    if line.strip().startswith('B'):
                        macro1.append(0.01 * float(line.strip().split(".")[3]))

    for inp_ in inp:
        time2, micro2, macro2 = [], [], []
        for i in range(1, 11):
            path = f"Figures_B/60/Figures_M_3/T{i}/{inp_}_{i}.txt"
            with open(path, 'r') as f:
                for line in f.readlines():
                    if line.strip().startswith("Training"):
                        time2.append(float(line.strip().split(":")[1]))
                    if line.strip().startswith("f1_score(micro)"):
                        micro2.append(float(line.strip().split(":")[1]))
                    if line.strip().startswith("f1_score(macro)"):
                        macro2.append(float(line.strip().split(":")[1]))

    t1_value , p1_value = ttest_ind(a=micro1, b= micro2, equal_var=False, alternative= 'greater')
    t2_value , p2_value = ttest_ind(a=macro1, b= macro2, equal_var=False, alternative= 'greater')
    t3_value, p3_value = ttest_ind(a=time1, b=time2, equal_var=False, alternative='greater')
    #print(micro1)
    #print(micro2)

    f1_micro_mean1, f1_micro_std1, f1_micro_var1 = stat.mean(micro1), stat.pstdev(micro1), stat.pvariance(micro1)
    f1_macro_mean1, f1_macro_std1, f1_macro_var1 = stat.mean(macro1), stat.pstdev(macro1), stat.pvariance(macro1)
    time_mean1, time_std1, time_var1 = stat.mean(time1), stat.pstdev(time1), stat.pvariance(time1)
    Upper_boundP_Mi = f1_micro_mean1+(t1_value*f1_micro_std1)
    lower_boundP_Mi = f1_micro_mean1-(t1_value * f1_micro_std1)
    Upper_boundP_Ma = f1_macro_mean1 + (t1_value * f1_macro_std1)
    lower_boundP_Ma = f1_macro_mean1 - (t1_value * f1_macro_std1)
    Upper_boundP_t =  time_mean1+(t3_value*time_std1)
    lower_boundP_t = time_mean1 - (t3_value * time_std1)
    #print(t2_value)
    #print(f1_macro_std1)

    f1_micro_mean2, f1_micro_std2, f1_micro_var2 = stat.mean(micro2), stat.pstdev(micro2), stat.pvariance(micro2)
    f1_macro_mean2, f1_macro_std2, f1_macro_var2 = stat.mean(macro2), stat.pstdev(macro2), stat.pvariance(macro2)
    Upper_bound_Mi = f1_micro_mean2+(t2_value*f1_micro_std2)
    lower_bound_Mi = f1_micro_mean2-(t2_value * f1_micro_std2)
    Upper_bound_Ma = f1_macro_mean2 + (t2_value * f1_macro_std2)
    lower_bound_Ma = f1_macro_mean2 - (t2_value * f1_macro_std2)
    time_mean2, time_std2, time_var2 = stat.mean(time2), stat.pstdev(time2), stat.pvariance(time2)
    #print(f"{inp_}_\tf1_micro: {f1_micro_mean:.2f} ({f1_micro_std:.2f}/{f1_micro_var:.2f})\tf1_macro: {f1_macro_mean:.2f} ({f1_macro_std:.2f}/{f1_macro_var:.2f})")
    #print()
    f1_micro_mean1=f1_micro_mean1*100
    f1_macro_mean1=f1_macro_mean1*100
    f1_micro_mean2=f1_micro_mean2*100
    f1_macro_mean2=f1_macro_mean2*100
    Upper_bound_t = time_mean2 + (t3_value * time_std2)
    lower_bound_t = time_mean2 - (t3_value * time_std2)


    print(f"f1_micro_P: {f1_micro_mean1:.2f} ({(f1_micro_mean1+Upper_boundP_Mi):.2f}-{f1_micro_mean1-lower_boundP_Mi:.2f}) \tf1_macro_P: {f1_macro_mean1:.2f} ({f1_macro_mean1+Upper_boundP_Ma:.2f}-{f1_macro_mean1-lower_boundP_Ma:.2f}) \t P1-Value: {p1_value:.6f} \t P2-Value: {p2_value:.6f} \ttime_Avg1: {time_mean1:.2f} ({time_mean1+Upper_boundP_t:.2f}-{time_mean1-lower_boundP_t:.2f}) ")
    print(f"f1_micro: {f1_micro_mean2:.2f} ({f1_micro_mean2+Upper_bound_Mi:.2f}-{f1_micro_mean2-lower_bound_Mi:.2f}) \tf1_macro: {f1_macro_mean2:.2f} ({f1_macro_mean2+Upper_bound_Ma:.2f}-{f1_macro_mean2-lower_bound_Ma:.2f}) \ttime_Avg2: {time_mean2:.2f} ({time_mean2+Upper_bound_t:.2f}-{time_mean2-lower_bound_t:.2f})")


if __name__ == '__main__':
    non_mem_utils()