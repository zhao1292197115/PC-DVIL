import os

# 这是你报错信息里显示的 DINOv2 本地源码绝对路径
repo_dir = "/home/d510/cobot_magic/dinov2_local/dinov2-main/dinov2"

print("🛠️ 开始为 DINOv2 打 Python 3.8 兼容性补丁...")

fixed_files = 0
# 遍历 DINOv2 所有的源代码文件
for root, dirs, files in os.walk(repo_dir):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 只要发现 Python 3.10 的高级类型提示，就一键降级
            if " | None" in content:
                content = content.replace("float | None", "float")
                content = content.replace("int | None", "int")
                content = content.replace("Tensor | None", "Any")
                content = content.replace("bool | None", "bool")
                content = content.replace("tuple | None", "tuple")
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"  ✅ 成功修复文件: {file}")
                fixed_files += 1

print(f"\n🎉 补丁打完啦！总共修复了 {fixed_files} 个文件。")
print("🚀 现在请直接去运行你的 python act/train.py 命令！")
