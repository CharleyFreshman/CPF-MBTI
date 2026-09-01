import os
MODEL_DIR = 'models/'
save_dir = f'{MODEL_DIR}two_parent'
tag = os.path.basename(save_dir.rstrip('/'))
print(f"{'='*5}[{tag}] 训练标签: I/E {'='*5}")
