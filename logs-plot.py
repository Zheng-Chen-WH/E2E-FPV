import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def one_line(): #对一个csv文件画
    smooth_factor=10000 #对列表中多少位数求平均，该值越大输出图像越平滑
    # 读取CSV文件
    csv_file_path1 = 'csv.csv'  # 替换为第一个CSV文件的路径
    df1 = pd.read_csv(csv_file_path1, header=None, skiprows=1)

    # 提取第一个CSV文件的x和y轴数据
    x_data1 = df1.iloc[500:50000, 1].astype(float)  # 调整数据绘图范围
    y_data1 = df1.iloc[500:50000, 2].astype(float)
    y_data2 = []
    for i in range(len(y_data1)-1):
        if i==0:
            y_data2.append(y_data1.iloc[0])
        if i <=smooth_factor and i>0:
            print(sum(y_data1.iloc[0:i])/i)
            y_data2.append(sum(y_data1.iloc[0:i])/i)
        else:
            y_data2.append(sum(y_data1.iloc[i-smooth_factor:i])/smooth_factor)
    # 读取第二个CSV文件
    # csv_file_path2 = 'train_loss.csv'  # 替换为第二个CSV文件的路径
    # df2 = pd.read_csv(csv_file_path2, header=None, skiprows=1)

    # 提取第二个CSV文件的x和y轴数据
    # x_data2 = df2.iloc[1:, 1].astype(float)
    # y_data2 = df2.iloc[1:, 2].astype(float)

    # 设置绘图风格，使用科学论文常见的线条样式和颜色
    plt.style.use('seaborn-v0_8-whitegrid')

    # 设置字体和字号
    font = {'family': 'serif',
            'serif': 'Times New Roman',
            'weight': 'normal',
            'size': 15,
            }
    plt.rc('font', **font)

    # 绘制第一幅图像
    plt.figure(1)
    plt.plot(x_data1, y_data1,color='blue', linewidth=0.5, label="loss")
    plt.plot(x_data1, y_data2,'r',linewidth=1, label="smoothed loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('policy loss-epochs')
    plt.legend()
    plt.tight_layout()
    # 调整布局使得图像不溢出
    plt.savefig('test_loss.svg', format='svg', bbox_inches='tight')

    # 绘制第二幅图像
    # plt.figure(2)
    # plt.plot(x_data2, y_data2, color='red', linewidth=2)
    # plt.xlabel('epochs')
    # plt.ylabel('train_loss')
    # plt.title('train_loss')

    plt.tight_layout()
    # 调整布局使得图像不溢出
    plt.savefig('test_reward.svg', format='svg', bbox_inches='tight')

    # 显示图形
    plt.show()

def one_line_with_fluctuation():
    smooth_factor=1000 #对列表中多少位数求平均，该值越大输出图像越平滑
    csv_file_path1 = 'sac_1.csv'  # 替换为第一个CSV文件的路径
    df1 = pd.read_csv(csv_file_path1, header=None, skiprows=1)
    csv_file_path2 = 'sac_2.csv'  # 替换为第一个CSV文件的路径
    df2 = pd.read_csv(csv_file_path2, header=None, skiprows=1)
    csv_file_path3 = 'sac_3.csv'  # 替换为第一个CSV文件的路径
    df3 = pd.read_csv(csv_file_path3, header=None, skiprows=1)
    csv_file_path4 = 'sac_4.csv'  # 替换为第一个CSV文件的路径
    df4 = pd.read_csv(csv_file_path4, header=None, skiprows=1)
    csv_file_path5 = 'sac_5.csv'  # 替换为第一个CSV文件的路径
    df5 = pd.read_csv(csv_file_path5, header=None, skiprows=1)

    # 提取第一个CSV文件的x和y轴数据
    x_data1 = df1.iloc[1:, 1].astype(float)
    y_data1 = df1.iloc[1:, 2].astype(float)
    y_data1_avg = []
    for i in range(len(y_data1)-1):
        if i==0:
            y_data1_avg.append(y_data1.iloc[0])
        if i <=smooth_factor and i>0:
            y_data1_avg.append(sum(y_data1.iloc[0:i])/i)
        else:
            y_data1_avg.append(sum(y_data1.iloc[i-smooth_factor:i])/smooth_factor)

    x_data2 = df2.iloc[1:, 1].astype(float)
    y_data2 = df2.iloc[1:, 2].astype(float)
    y_data2_avg = []
    for i in range(len(y_data2)-1):
        if i==0:
            y_data2_avg.append(y_data2.iloc[0])
        if i <=smooth_factor and i>0:
            y_data2_avg.append(sum(y_data2.iloc[0:i])/i)
        else:
            y_data2_avg.append(sum(y_data2.iloc[i-smooth_factor:i])/smooth_factor)

    x_data3 = df3.iloc[1:, 1].astype(float)
    y_data3 = df3.iloc[1:, 2].astype(float)
    y_data3_avg = []
    for i in range(len(y_data3)-1):
        if i==0:
            y_data3_avg.append(y_data3.iloc[0])
        if i <=smooth_factor and i>0:
            y_data3_avg.append(sum(y_data3.iloc[0:i])/i)
        else:
            y_data3_avg.append(sum(y_data3.iloc[i-smooth_factor:i])/smooth_factor)

    x_data4 = df4.iloc[1:, 1].astype(float)
    y_data4 = df4.iloc[1:, 2].astype(float)
    y_data4_avg = []
    for i in range(len(y_data4)-1):
        if i==0:
            y_data4_avg.append(y_data4.iloc[0])
        if i <=smooth_factor and i>0:
            y_data4_avg.append(sum(y_data4.iloc[0:i])/i)
        else:
            y_data4_avg.append(sum(y_data4.iloc[i-smooth_factor:i])/smooth_factor)

    x_data5 = df5.iloc[1:, 1].astype(float)
    y_data5 = df5.iloc[1:, 2].astype(float)
    y_data5_avg = []
    for i in range(len(y_data5)-1):
        if i==0:
            y_data5_avg.append(y_data5.iloc[0])
        if i <=smooth_factor and i>0:
            y_data5_avg.append(sum(y_data5.iloc[0:i])/i)
        else:
            y_data5_avg.append(sum(y_data5.iloc[i-smooth_factor:i])/smooth_factor)
    
    min_length = min(len(arr) for arr in [y_data1_avg, y_data2_avg, y_data3_avg, y_data4_avg, y_data5_avg])
    # 使用切片裁剪每个NumPy数组
    y_data1_avg = y_data1_avg[:min_length]
    y_data2_avg = y_data2_avg[:min_length]
    y_data3_avg = y_data3_avg[:min_length]
    y_data4_avg = y_data4_avg[:min_length]
    y_data5_avg = y_data5_avg[:min_length]
    x_data = x_data1[:min_length]
    y_avg_min=[]
    y_avg_max=[]
    y_avg=[]
    for i in range(min_length):
        y_avg_min.append(min(y_data1_avg[i],y_data2_avg[i],y_data3_avg[i],y_data4_avg[i],y_data5_avg[i]))
        y_avg_max.append(max(y_data1_avg[i],y_data2_avg[i],y_data3_avg[i],y_data4_avg[i],y_data5_avg[i]))
        y_avg.append((sum([y_data1_avg[i],y_data2_avg[i],y_data3_avg[i],y_data4_avg[i],y_data5_avg[i]]))/5)

    # 设置绘图风格，使用科学论文常见的线条样式和颜色
    plt.style.use('seaborn-whitegrid')

    # 设置字体和字号
    font = {'family': 'serif',
            'serif': 'Times New Roman',
            'weight': 'normal',
            'size': 15,
            }
    plt.rc('font', **font)

    # 绘制第一幅图像
    plt.figure(1)
    plt.plot(x_data, y_avg,color='red', linewidth=1, label="average reward")
    plt.fill_between(x_data, y1=y_avg_min,y2=y_avg_max,where=np.arange(len(x_data)),color='r', alpha=0.2) #做出reward图中的波动效果，alpha参数控制填充颜色的透明度
    plt.xlabel('epochs')
    plt.ylabel('reward')
    plt.title('SAC reward-epochs')
    plt.legend()
    plt.tight_layout()
    # 调整布局使得图像不溢出
    plt.savefig('SAC_reward.svg', format='svg', bbox_inches='tight')

    # 显示图形
    plt.show()

class multi_training_process():
    def __init__(self,file_list):
        self.min_len=999999
        self.file_list=file_list
        self.smooth_factor=1000
        for file_name in file_list:
            df = pd.read_csv(file_name, header=None, skiprows=1)
            x_data = df.iloc[1:, 1].astype(float)
            data_len=len(df.iloc[1:, 1].astype(float))
            if data_len<self.min_len:
                self.min_len=data_len
        self.x_data=x_data[:self.min_len]
    def calcu_avg(self,file_name):
        df = pd.read_csv(file_name, header=None, skiprows=1)
        # 提取CSV文件的x和y轴数据
        y_data = df.iloc[1:, 2].astype(float)
        y_data = y_data[:self.min_len]
        y_data_avg = []
        for i in range(len(y_data)-1):
            if i==0:
                y_data_avg.append(y_data.iloc[0])
            if i <=self.smooth_factor and i>0:
                y_data_avg.append(sum(y_data.iloc[0:i])/i)
            else:
                y_data_avg.append(sum(y_data.iloc[i-self.smooth_factor:i])/self.smooth_factor)
        return y_data_avg
    def fluctuation(self):
        y_list=[]
        y_avg_min=[]
        y_avg_max=[]
        y_avg_list=[]
        print(self.file_list)
        for file in self.file_list:
            y_list.append(self.calcu_avg(file))
        for i in range(self.min_len):     
            y_min=99999999
            for j in range(len(y_list)):
                if y_list[j][i]<=y_min:
                    y_min=y_list[j][i]
            y_avg_min.append(y_min)
            y_max=-999999999
            for j in range(len(y_list)):
                if y_list[j][i]>=y_max:
                    y_max=y_list[j][i]
            y_avg_max.append(y_max)
            y_avg=0
            for j in range(len(y_list)):
                y_avg+=y_list[j][i]
            y_avg_list.append(y_avg/len(y_list))
        return self.x_data,y_avg_min,y_avg_max,y_avg_list

#one_line_with_fluctuation()

# files1=["final_1_sr.csv","final_2_sr.csv","final_3_sr.csv"]
# plot1=multi_training_process(files1)
# x_final,y_min_final,y_max_final,y_avg_final=plot1.fluctuation()
# files2=["future_1_sr.csv","future_2_sr.csv"]
# plot2=multi_training_process(files2)
# x_future,y_min_future,y_max_future,y_avg_future=plot2.fluctuation()
# files3=["final_1_mr.csv","final_2_mr.csv","final_3_mr.csv"]
# plot3=multi_training_process(files3)
# x_final_mr,y_min_final_mr,y_max_final_mr,y_avg_final_mr=plot3.fluctuation()
# files4=["future_1_mr.csv","future_2_mr.csv"]
# plot4=multi_training_process(files4)
# x_future_mr,y_min_future_mr,y_max_future_mr,y_avg_future_mr=plot4.fluctuation()
# # least_len=np.min(len(x_ppo),len(x_sac))
# # x_sac=x_sac[:least_len]
# # y_min_sac=y_min_sac[:least_len]
# # y_max_sac=y_max_sac[:least_len]
# # y_avg_sac=y_avg_sac[:least_len]
# # x_ppo=x_sac[:least_len]
# # y_min_ppo=y_min_ppo[:least_len]
# # y_max_ppo=y_max_ppo[:least_len]
# # y_avg_ppo=y_avg_ppo[least_len]
# # 设置绘图风格，使用科学论文常见的线条样式和颜色
# plt.style.use('seaborn-whitegrid')

# # 设置字体和字号
# font = {'family': 'serif',
#         'serif': 'Times New Roman',
#         'weight': 'normal',
#         'size': 15,
#         }
# plt.rc('font', **font)

# # 绘制第一幅图像
# plt.figure(1)
# plt.plot(x_final, y_avg_final,color='blue', linewidth=1, label="final mode")
# plt.fill_between(x_final, y1=y_min_final,y2=y_max_final,where=np.arange(len(x_final)),color='blue', alpha=0.2) #做出reward图中的波动效果，alpha参数控制填充颜色的透明度
# plt.plot(x_future, y_avg_future,color='red', linewidth=1, label="future mode")
# plt.fill_between(x_future, y1=y_min_future,y2=y_max_future,where=np.arange(len(x_future)),color='red', alpha=0.2) #做出reward图中的波动效果，alpha参数控制填充颜色的透明度
# plt.plot(x_final_mr, y_avg_final_mr,color='orange', linewidth=1, label="final mode without shaping")
# plt.fill_between(x_final_mr, y1=y_min_final_mr,y2=y_max_final_mr,where=np.arange(len(x_final_mr)),color='orange', alpha=0.2)
# plt.plot(x_future_mr, y_avg_future_mr,color='green', linewidth=1, label="future mode without shaping")
# plt.fill_between(x_future_mr, y1=y_min_future_mr,y2=y_max_future_mr,where=np.arange(len(x_future_mr)),color='green', alpha=0.2)
# plt.xlabel('episodes')
# plt.ylabel('reward')
# plt.title('Effects of shaping reward')
# plt.legend()
# plt.tight_layout()
# # 调整布局使得图像不溢出
# plt.savefig('shaping reward comparison.svg', format='svg', bbox_inches='tight')

# # 显示图形
# plt.show()

one_line()