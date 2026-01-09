import torch
import smplx
import sys


def calculate_smplx_height(model_folder_path, gender='neutral'):
    """
    计算 SMPL-X 模型在 betas 全 0 时的身高 (顶点 Y 轴最大值 - 最小值)
    """
    try:
        # 1. 创建 SMPL-X 模型
        # num_betas=10 是常用的形状参数数量
        model = smplx.create(
            model_path=model_folder_path,
            model_type='smplx',
            gender=gender,
            use_face_contour=False,
            num_betas=10,
            ext='npz'  # 或者 'pkl'，取决于你下载的文件格式
        )
    except FileNotFoundError:
        print(f"错误: 在路径 '{model_folder_path}' 下找不到模型文件。")
        print("请确保你已下载 SMPLX_{GENDER}.npz 并指定正确的文件夹路径。")
        return

    # 2. 设置 Betas 为全 0
    # shape: [batch_size, num_betas]
    betas = torch.zeros(1, 10)

    # 3. 设置 body_pose 为全 0 (T-pose / Mean pose)
    # 这一步是可选的，因为如果不传，默认就是 0，但显式写出来更严谨
    body_pose = torch.zeros(1, 21 * 3)

    # 4. 前向传播获取顶点
    # 不传入 expression (表情) 等其他参数，默认也为 0
    output = model(
        betas=betas,
        body_pose=body_pose,
        return_verts=True
    )

    vertices = output.vertices[0]  # 取出第一个 batch, shape: [10475, 3]

    # 5. 计算身高
    # SMPL-X 的坐标系通常 Y 轴是向上的 (Up axis)
    # vertices[:, 0] -> X (左右)
    # vertices[:, 1] -> Y (上下)
    # vertices[:, 2] -> Z (前后)

    max_y = torch.max(vertices[:, 1])
    min_y = torch.min(vertices[:, 1])

    height = max_y - min_y

    # 打印详细信息
    print(f"--- 模型信息 ({gender}) ---")
    print(f"最高点 (Y max): {max_y.item():.4f} m")
    print(f"最低点 (Y min): {min_y.item():.4f} m")
    print(f"计算身高 (Max - Min): {height.item():.4f} m")

    return height.item()


# ================= 使用示例 =================
# 请将此路径修改为你存放 SMPLX_NEUTRAL.npz 的文件夹路径
# 例如: './models/' 或 '/home/user/smplx_models/'
MODEL_FOLDER_PATH = '/home/miku/Documents/VideoSkills/data/SMPL'

if __name__ == "__main__":
    # 如果没有 GPU 也没关系，默认是在 CPU 上运行的
    with torch.no_grad():
        h = calculate_smplx_height(MODEL_FOLDER_PATH, gender='neutral')