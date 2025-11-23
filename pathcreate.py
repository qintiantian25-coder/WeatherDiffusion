import os

# --- 配置参数 ---
# 数据所在的主文件夹（相对于项目根目录）
data_root_dir = 'data'
# 您的图像数据所在子文件夹名
custom_data_folder = 'custom_data'
# 列表文件要放置的目录和文件名
list_file_path = os.path.join(data_root_dir, 'allweather', 'validation.txt')
# 包含训练数据的子目录
training_sub_dir = 'val'

# 构造图像输入文件夹的绝对路径，用于读取文件名
input_images_path = os.path.join(os.getcwd(), data_root_dir, custom_data_folder, training_sub_dir, 'image')
# -----------------

# 确保列表文件所在的目录存在
os.makedirs(os.path.dirname(list_file_path), exist_ok=True)

# 列表文件内容所需的相对路径前缀
# 列表文件在 'allweather' 下，需要先跳出到 'data'，再进入 'custom_data'
relative_prefix = os.path.join('..', custom_data_folder)

print(f"开始扫描图像目录：{input_images_path}")
print(f"列表文件将写入：{list_file_path}")
print(f"列表内容使用的前缀：{relative_prefix}/...")

count = 0
with open(list_file_path, 'w') as f:
    try:
        # 获取所有输入图像文件名，并排序以保证 Input 和 Label 对应
        for filename in sorted(os.listdir(input_images_path)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 构造相对于列表文件所在位置的 Input 和 Label 路径
                input_rel_path = os.path.join(relative_prefix, training_sub_dir, 'image', filename).replace(os.sep, '/')
                label_rel_path = os.path.join(relative_prefix, training_sub_dir, 'label', filename).replace(os.sep, '/')

                # 写入：Input_Path Label_Path
                f.write(f"{input_rel_path} {label_rel_path}\n")
                count += 1
    except FileNotFoundError:
        print("\n❌ 错误：无法找到输入图像目录！请检查您的 'data/custom_data/train/image' 路径是否正确。")
        exit()

print(f"\n✅ 列表文件生成成功！共写入 {count} 对图像路径。")