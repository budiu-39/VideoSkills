# VideoSkills

端到端的 pipeline：通过输入带有人类的运动视频，控制机器人做出相同的动作。  


![Demo](demo/dance.gif)
*Unitree G1 通过视频学习如何跳舞*
![Demo](demo/gym.gif)
*运动数据去噪功能：相较于Motion Estimator的输出（红色人物角色），去噪后的结果（绿色人物）更加稳定和真实*

## 功能特性

| 功能                               | 支持情况 |
|------------------------------------|----------|
| 训练通用运动模型                   | ✅       |
| 训练针对单个/多个运动的专家模型     | ✅       |
| 运动重定向                         | ✅       |
| 运动数据去噪/改善                       | ✅       |
| 通过语言控制运动                   | 🔜 Coming soon |


## 支持的机器人平台

| 平台        | 支持情况   |
|-------------|------------|
| SMPL robot  | ✅          |
| Unitree G1  | ✅          |
| Unitree H1  | 🔜 Coming soon |

---

## Setup

```bash
# 创建并激活环境
conda create -n isaac python=3.8 -y
conda activate isaac

# 安装依赖模块
for d in . rsl_rl GVHMR isaacgym/python; do
  (cd "$d" && pip install -e .)
done

pip install -r requirement.txt
````

⚠️ 注意：安装 `numpy` 时可能会报不兼容警告，但**不影响使用**。

### 下载代码与模型

```bash
git clone git@github.com:budiu-39/VideoSkills.git
hf download Budiu39/VideoSkills --local-dir ./VideoSkills
```

---

## Refine

运行 refine 代码（指定视频文件夹位置为 `--folder` 参数）。

### SMPL robot

（总共 2 个运动片段，预计需要 5–10 分钟）

```bash
python videoskills/refine.py --task=smpl --folder=demo/test_2 --static_cam --headless --accelerate
```

### G1 robot

（总共 2 个运动片段，预计需要 5–10 分钟）

```bash
python videoskills/refine.py --task=g1 --folder=demo/test_2 --static_cam --headless --accelerate
```

输出结果：

* `logs/smpl_ppo/<对应 run>/refine_results` 内为 refine 结果
* `refine_results/renders` 内为渲染视频

---

## 训练 PHC 模型

### G1

```bash
python videoskills/train.py --task=g1
```

### SMPL

```bash
python videoskills/train.py --task=smpl
```

