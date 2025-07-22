# yoData
yolo_data_argument

# 目标检测数据增强工具

## 1. 简介

本项目是一个为目标检测任务设计的数据增强工具。在深度学习模型（尤其是目标检测）的训练过程中，充足且多样化的数据集是提升模型性能和泛化能力的关键。当原始数据集规模有限时，模型很容易出现过拟合，即在训练集上表现优异，但在未见过的测试数据上表现不佳。

数据增强是一种经济高效的解决方案。它通过对现有数据进行一系列随机变换（如旋转、裁剪、调整亮度等）来生成新的、合成的训练样本。这不仅极大地扩充了数据集的规模，更重要的是增加了数据的多样性，模拟了真实世界中可能出现的各种场景。

本工具旨在自动化这一过程，帮助用户从少量原始图像和YOLO格式的标签中，快速生成大量增强后的数据，从而有效地扩充训练集。

## 2. 项目优势与潜在风险

### ✨ 优点 (Advantages)

*   **提升模型泛化能力**: 通过生成多样化的图像，模型能够学习到更鲁棒的特征，从而更好地适应新环境。
*   **抑制过拟合**: 扩充数据集是解决过拟合最有效的方法之一。本工具可以轻松将数据集规模扩大数十倍。
*   **降低数据成本**: 无需耗费大量人力和物力去采集和标注新的真实世界图像。
*   **提升模型鲁棒性**: 模拟了光照变化、物体平移、视角变化等情况，使模型在复杂条件下表现更稳定。

### ⚠️ 潜在隐患 (Potential Risks)

*   **引入噪声标签**: 不恰当的增强（如过度旋转或裁剪）可能导致边界框（bounding box）无法准确包裹目标物体，从而引入错误的标签信息。
*   **改变数据分布**: 过度的增强可能会生成与真实世界差异过大的失真图像，改变原始数据的分布，反而误导模型的学习。
*   **参数依赖性**: 数据增强的效果高度依赖于参数设置。需要根据具体任务仔细调整每种增强方法的概率和强度，建议在生成后对部分样本进行抽样检查。

## 3. 功能特性

本脚本实现了以下常见的目标检测数据增强方法：

*   **几何变换**
    *   **裁剪 (Crop)**: 随机裁剪图像，同时保证所有目标物体完整出现在裁剪后的区域内。
    *   **平移 (Shift)**: 在不将目标移出视野的前提下，随机小幅度平移图像。
    *   **旋转 (Rotate)**: 在指定角度范围内随机旋转图像。
    *   **镜像 (Flip)**: 对图像进行水平、垂直或对角翻转。
*   **像素级变换**
    *   **亮度调整 (Change Light)**: 随机改变图像的亮度。
    *   **增加噪声 (Add Noise)**: 向图像中加入高斯噪声，模拟传感器噪声。
    *   **Cutout**: 随机在图像中遮挡一个或多个矩形区域，强迫模型关注物体的全局信息而非局部细节。

**核心功能**: 所有几何变换都会**自动、精确地调整**对应的YOLO格式边界框，确保标签与增强后图像的匹配。

## 4. 如何使用

### 4.1. 环境配置

首先，请确保您已安装所需的Python库。

```bash
pip install numpy opencv-python scikit-image
```

### 4.2. 目录结构

建议您的项目遵循以下目录结构：

```
/your_project
|-- source_data/
|   |-- images/         # 存放原始图片 (e.g., a.jpg, b.png)
|   |-- labels/         # 存放原始YOLO标签 (e.g., a.txt, b.txt)
|-- datasets/
|   |-- Images/         # (脚本自动创建) 用于存放增强后的图片
|   |-- labels/         # (脚本自动创建) 用于存放增强后的标签
|-- augment.py          # 本脚本文件
```

### 4.3. 运行脚本

您可以通过命令行参数来指定输入和输出路径。

```bash
python augment.py --source_img_path /path/to/your/source/images \
                  --source_txt_path /path/to/your/source/labels \
                  --save_img_path /path/to/your/augmented/Images \
                  --save_txt_path /path/to/your/augmented/labels
```

**例如**:

```bash
python augment.py --source_img_path ./source_data/images \
                  --source_txt_path ./source_data/labels \
                  --save_img_path ./datasets/Images \
                  --save_txt_path ./datasets/labels
```

### 4.4. 参数自定义

您可以直接在脚本的 `if __name__ == '__main__':` 部分修改以下核心参数：

*   `need_aug_num`: 每张原始图片希望生成的增强样本数量。
    ```python
    need_aug_num = 20  # 每张图片生成20个增强版本
    ```

*   **数据增强策略**: 在 `DataAugmentForObjectDetection` 类的初始化方法中，您可以精细地控制每种增强方法的触发概率和相关属性。

    ```python
    # 在 main 函数中
    dataAug = DataAugmentForObjectDetection(
        rotation_rate=0.5,       # 50% 的概率进行旋转
        max_rotation_angle=10,   # 最大旋转角度为 10 度
        shift_rate=0.5,          # 50% 的概率进行平移
        add_noise_rate=0.3,      # 30% 的概率增加噪声
        change_light_rate=0.6,   # 60% 的概率改变亮度
        flip_rate=0.5,           # 50% 的概率进行镜像
        cutout_rate=0.4,         # 40% 的概率进行Cutout
        cut_out_length=40,       # Cutout区域的边长
        is_cutout=True           # 启用Cutout
    )
    ```

## 5. 代码结构解析

*   **`DataAugmentForObjectDetection` 类**:
    *   封装了所有数据增强的核心逻辑。
    *   每个增强方法（如 `_rotate_img_bbox`）都同时处理图像和边界框（bboxes），确保二者的同步更新。
    *   `dataAugment` 方法是总入口，它会根据设定的概率随机应用一种或多种增强效果。

*   **`ToolHelper` 类**:
    *   一个辅助工具类，负责文件I/O操作。
    *   `parse_yolo_txt()`: 解析YOLO格式的 `.txt` 标签文件，将其从归一化坐标转换为绝对像素坐标 `[x_min, y_min, x_max, y_max]`。
    *   `save_yolo_txt()`: 将增强后的绝对像素坐标边界框转换回YOLO的归一化格式并保存为 `.txt` 文件。
    *   `save_img()`: 保存增强后的图像。

*   **`main` 主函数**:
    *   负责解析命令行参数。
    *   遍历所有源图像，为每张图像循环调用数据增强流程，并保存结果。
