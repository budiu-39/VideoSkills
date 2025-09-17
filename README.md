# VideoSkills

这是一个端到端的流程，输入带有人类运动的视频，可以控制机器人做出相同的动作，兼容多种机器人平台。目前已经实现了仿真部分。  

## 功能特性

| 功能                               | 支持情况 |
|------------------------------------|----------|
| 训练通用运动模型                   | ✅       |
| 训练针对单个/多个运动的专家模型     | ✅       |
| 运动重定向                         | ✅       |
| 运动数据去噪/改善                  | ✅       |
| 通过语言控制运动                   | 🔜 Coming soon |
| 部署到真实的机器人上               | 🔜 Coming soon |

![Demo](demo/dance.gif)  
Unitree G1 通过视频学习如何跳舞
![Demo](demo/gym.gif)  
运动数据去噪：Motion Estimator原始输出（红色人物角色），去噪后的结果（绿色人物）



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

# 下载代码与模型（提供在SMPL和G1机器人平台上训练好的PHC模型）

git clone git@github.com:budiu-39/VideoSkills.git
hf download Budiu39/VideoSkills --local-dir ./VideoSkills

# 安装依赖模块
for d in . rsl_rl GVHMR isaacgym/python; do
  (cd "$d" && pip install -e .)
done



pip install -r requirement.txt
```

⚠️ 注意：安装 `numpy` 时可能会报不兼容警告，但**不影响使用**。



## 使用说明
运行以下代码，通过 `--task` 指定 机器人平台（smpl/g1),  `--folder` 指定视频。

```bash

# SMPL robot （总共 2 个运动片段，预计需要 5–10 分钟）
python videoskills/refine.py --task=smpl --folder=demo/test_2 --static_cam --headless --accelerate


# G1 robot 
python videoskills/refine.py --task=g1 --folder=demo/test_2 --static_cam --headless --accelerate
```


输出结果在 `logs/<smpl/g1>_ppo/<run_name>`内，`gvhmr_results` 为 Motion Estimator 预测的运动，pt 文件为训练得到的机器人控制模型， `renders_results` 为去噪后的运动结果， `renders_results` 文件夹内为渲染好的对比视频。

---

## 训练 PHC 模型（通用模型）

### G1

```bash
python videoskills/train.py --task=g1
```

### SMPL

```bash
python videoskills/train.py --task=smpl
```

