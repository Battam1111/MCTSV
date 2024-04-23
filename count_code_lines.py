import os

def count_py_lines(start_path, excluded_dirs):
    total_lines = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        # 过滤出需要排除的目录
        dirnames[:] = [d for d in dirnames if os.path.join(dirpath, d) not in excluded_dirs]
        for filename in filenames:
            if filename.endswith('.py'):
                file_path = os.path.join(dirpath, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    num_lines = sum(1 for line in file)
                    total_lines += num_lines
                    print(f"{file_path}: {num_lines} lines")
    return total_lines

if __name__ == '__main__':
    current_directory = os.getcwd()
    # 定义要排除的目录列表
    excluded_directories = [os.path.join(current_directory, 'mctsv')]
    total_lines = count_py_lines(current_directory, excluded_directories)
    print(f"Total number of Python code lines in directory and subdirectories: {total_lines}")
