from line_profiler import LineProfiler, show_text
from line_profiler import load_stats  # 正确导入 load_stats

# 加载 .lprof 文件
with open('profile_output.lprof', 'rb') as f:
    stats = load_stats(f.read())  # 使用 load_stats 函数来加载数据

# 创建一个新的 LineProfiler 实例来显示统计数据
profiler = LineProfiler()
show_text(stats, profiler.code_map)