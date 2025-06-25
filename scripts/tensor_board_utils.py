from tensorboard.backend.event_processing import event_accumulator

# 加载路径
event_path = "/home/miku/Documents/VideoSkills/events.out.tfevents.1749681523.galvani-cn207"
ea = event_accumulator.EventAccumulator(event_path)
ea.Reload()

# 获取所有 scalar tags
scalar_tags = ea.Tags()["scalars"]
print("Available scalar tags:", scalar_tags)

# 读取某个 tag 的所有记录
for tag in scalar_tags:
    # print(f"\n== {tag} ==")
    # for event in ea.Scalars(tag)[:5]:  # 打印前5个数据点
    #     print(f"Step {event.step}, Value {event.value}")
    if tag == "eval/success_rate" or  tag ==  "eval/mpjpe_all" or tag == "eval/mpjpe_succ":
        print(f"\n== {tag} ==")
        for event in ea.Scalars(tag):
            print(f"Step {event.step}: Value {event.value}")