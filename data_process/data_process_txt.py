# 将txt中的格式文件修改为数字+文件名后缀
# 例如：Coverage数据集文件里面的fake.txt文件
# /user/Jfu/cvl/semi_supervise_local/2025/10_11_SCCM/Dataset/finetune_dataset/Coverage/image/29t.tif修改为29t.tif
input_file = r"/root/HiFi_IFDL-main/data_dir/Coverage/fake.txt"
lines = []

with open(input_file, "r") as f:
    for line in f:
        filename = line.strip().split("/")[-1]
        lines.append(filename + "\n")

with open(input_file, "w") as f:
    f.writelines(lines)

print("Coverage/fake.txt原文件已覆盖完成！")

input_file = r"/root/HiFi_IFDL-main/data_dir/columbia/vallist.txt"
lines = []

with open(input_file, "r") as f:
    for line in f:
        filename = line.strip().split("/")[-1]
        lines.append(filename + "\n")

with open(input_file, "w") as f:
    f.writelines(lines)

print("columbia/vallist.txt原文件已覆盖完成！")

