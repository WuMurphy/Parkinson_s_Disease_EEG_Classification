# 这是一个示例 Python 脚本。

# 按 ⌃R 执行或将其替换为您的代码。
# 按 双击 ⇧ 在所有地方搜索类、文件、工具窗口、操作和设置。
from lime import submodular_pick
import numpy as np

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    data = np.load("./UNMData/EyesClose_AllChannels_AllFeatures/train/801_PD.npy", allow_pickle=True).item()
    data = data["source"]
    pass

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
