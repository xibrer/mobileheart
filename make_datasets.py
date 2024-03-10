import os

from scipy import io


def slide_window_and_save(acc_result, window_size, step, file, save_dir):
    """
    滑动窗口并保存结果
    :param acc_result: 输入数据
    :param window_size: 窗口大小
    :param step: 步长
    :param file: 文件名
    :param save_dir: 保存目录
    :return: 无
    """
    d_x, d_y, d_z, d_noise = acc_result[0], acc_result[1], acc_result[2], acc_result[2]
    for i in range(0, len(d_y) - window_size + 1, step):
        x = d_x[i:i + window_size]
        y = d_y[i:i + window_size]
        z = d_z[i:i + window_size]
        noise = d_noise[i:i + window_size]

        # 创建新的文件名，由原始文件名（去掉扩展名）和当前窗口的起始索引组成
        file_name = os.path.join(save_dir, f"{file.split('.')[0]}_{i}.mat")

        io.savemat(file_name, {'x': x, 'y': y, 'z': z, 'noise': noise})


def process_data(data_dir, window_size: int = 640, step: int = 640):
    """
    处理数据
    :param data_dir: 数据目录
    :param window_size: 窗口大小
    :param step: 步长
    :return: 无
    """
    for user in os.listdir(data_dir):
        user_folder = os.path.join(data_dir, user)
        save_dir = user_folder.replace('_collecting', 'sets')
        print(user_folder)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  # 如果目录不存在，创建目录
        for file in os.listdir(user_folder):
            # 加载.mat文件
            data = io.loadmat(os.path.join(user_folder, file))
            acc_result = data['accresult']
            slide_window_and_save(acc_result, window_size, step, file, save_dir)


if __name__ == '__main__':
    # path to process
    train_dir = 'data_collecting/standard_scg/'
    ecg_dir = 'data_collecting/ecg_scg/'
    scg_dir = 'data_collecting/scg/'
    test_dir = 'data_collecting/motion_scg/'
    noise_dir = 'data_collecting/noise/'

    # process test set
    process_data(noise_dir, step=80)
    process_data(train_dir, step=20)
    process_data(test_dir, step=640)

    print('finish')
