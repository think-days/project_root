# fix_encoding.py - 临时脚本，修复编码问题
import os
import codecs


def fix_file_encoding(filepath):
    """修复文件编码为UTF-8"""
    # 尝试不同的编码
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16', 'latin-1']

    content = None
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()
            print(f"Successfully read {filepath} with {encoding}")
            break
        except:
            continue

    if content:
        # 写回为UTF-8
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Converted {filepath} to UTF-8")
    else:
        print(f"Failed to read {filepath}")


# 修复项目中的所有Python文件
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            fix_file_encoding(filepath)
