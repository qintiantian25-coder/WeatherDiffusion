import os
import argparse


def generate_test_list(root_dir):
    """
    扫描测试目录，并生成 test_list.txt 文件。

    Args:
        root_dir (str): 测试数据集的根目录，即 'data/custom_data/test'。
    """

    input_dir = os.path.join(root_dir, 'image')
    gt_dir = os.path.join(root_dir, 'label')

    # 最终的 test_list.txt 文件将存放在 image 目录下
    output_file_path = os.path.join(input_dir, 'test_list.txt')

    # 检查两个目录是否存在
    if not os.path.isdir(input_dir):
        print(f"❌ 错误：未找到带噪图像目录: {input_dir}")
        return
    if not os.path.isdir(gt_dir):
        print(f"❌ 错误：未找到干净图像目录: {gt_dir}")
        return

    # 获取带噪图像（Input）的文件列表
    # os.listdir() 获取的是文件名，我们假设这些是列表的基础
    input_files = sorted(
        [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and not f.startswith('.')])

    # 过滤掉 image 目录下的 test_list.txt 文件本身
    if 'test_list.txt' in input_files:
        input_files.remove('test_list.txt')

    missing_gt_count = 0
    list_entries = []

    for input_filename in input_files:
        # 假设 GT 文件名与 Input 文件名相同
        gt_filename = input_filename

        # 检查对应的 GT 文件是否存在
        if os.path.exists(os.path.join(gt_dir, gt_filename)):
            # 构造相对路径：
            # Input 路径是相对于 input_dir 的文件名，即 input_filename
            # GT 路径是相对于 input_dir 的相对路径，即 '../label/' + gt_filename

            # 目标输出格式: <input文件名> <../label/gt文件名>
            gt_relative_path = os.path.join('..', 'label', gt_filename).replace('\\', '/')

            list_entries.append(f"{input_filename} {gt_relative_path}")
        else:
            print(f"⚠️ 警告：跳过 {input_filename}，因为未在 {gt_dir} 中找到对应的干净图像。")
            missing_gt_count += 1

    if not list_entries:
        print("❌ 错误：没有找到任何配对的图像文件。请检查文件名是否一致。")
        return

    # 写入文件
    with open(output_file_path, 'w') as f:
        f.write('\n'.join(list_entries))

    print("--------------------------------------------------")
    print(f"✅ 成功生成文件列表: {output_file_path}")
    print(f"✅ 共 {len(list_entries)} 对图像已记录。")
    if missing_gt_count > 0:
        print(f"⚠️ 警告：有 {missing_gt_count} 个输入图像缺少对应的干净图像。")
    print("--------------------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate test_list.txt for AllWeather Dataset.')
    parser.add_argument("--test_root", type=str,
                        default='data/custom_data/test',
                        help="Root directory of the test set (containing 'image' and 'label' folders).")
    args = parser.parse_args()

    generate_test_list(args.test_root)