import torch
import numpy as np

def get_embeddings(model, loader, scaler, device):
    """
    遍历数据，提取 Latent Code (Mu)
    使用 get_skill_embedding 加速推理
    """
    model.eval()
    embeddings = []

    with torch.no_grad():
        for batch in loader:
            # 1. 获取数据
            if isinstance(batch, dict):
                state = batch["state"]
            else:
                state = batch

            state = state.to(device)

            # 2. ★★★ 关键步骤：必须先归一化！★★★
            state_norm = scaler.transform_state(state)

            # 3. 调用加速接口 (只跑 Encoder)
            mu = model.get_skill_embedding(state_norm)

            embeddings.append(mu.cpu().numpy())

    return np.vstack(embeddings)

