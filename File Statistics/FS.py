import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")


countall = 0
countfolder = 0
def walkFile(file):
    listnum = []
    listnam = []
    for root, dirs, files in os.walk(file):
    # root 表示当前正在访问的文件夹路径
    # dirs 表示该文件夹下的子目录名list
    # files 表示该文件夹下的文件list
        num = 0

        # 遍历文件
        for f in files:
            global countall
            countall += 1
            num += 1
            #print(os.path.join(root, f))
            #print(num)
        listnum.append(num)

        # 遍历文件夹
        for d in dirs:
            global countfolder
            countfolder += 1
            #print(os.path.join(root, d))
            listnam.append(d)
    print("所有文件(文件+文件夹）数量一共为:", countall)
    print("文件夹数量一共为:", countfolder)
    print('文件数量一共为:', countall - countfolder)
    print(listnum)
    print(listnam)
    del(listnum[0])
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(listnam, listnum, alpha=0.8)
    plt.title("Number of Documents Statistics")
    plt.ylabel('# of items', fontsize=12)
    plt.xlabel('Name', fontsize=12)
    plt.show()


if __name__ == '__main__':
    walkFile(r"/Volumes/UNTITLED/fall11_whole")













