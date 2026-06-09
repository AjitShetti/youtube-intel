import os
import re

files_to_modify = [
    "main.py",
    "src/config.py",
    "src/scraper/youtube_scraper.py",
    "src/preprocessing/clean_text.py",
    "src/sentiment/sentiment_model.py",
    "src/embeddings/embedder.py",
    "src/topic_modeling/topic_model.py"
]

logger_import = "from src.utils.logger import setup_logger\nlogger = setup_logger(__name__)\n\n"

for filepath in files_to_modify:
    if not os.path.exists(filepath):
        continue
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # If already has logger, skip
    if "setup_logger" in content:
        continue

    # Insert logger import after the last import statement
    lines = content.split('\n')
    last_import_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            last_import_idx = i

    lines.insert(last_import_idx + 1, logger_import)
    content = '\n'.join(lines)

    # Regex replacements
    # Error
    content = re.sub(r'print\((f?["\'](?:\[ERROR\]|Error:)\s*)(.*?)\)', r'logger.error(\1\2)', content)
    # Warning
    content = re.sub(r'print\((f?["\'](?:\[WARNING\]|Warning:)\s*)(.*?)\)', r'logger.warning(\1\2)', content)
    # Info
    content = re.sub(r'print\((f?["\'](?:\[INFO\]|\[OK\])\s*)(.*?)\)', r'logger.info(\1\2)', content)
    # Generic Print -> logger.info
    content = re.sub(r'print\(', r'logger.info(', content)
    
    # Remove the literal [INFO], [ERROR], etc. from the logged strings if we want to be cleaner
    content = re.sub(r'logger\.error\((f?["\'])\[ERROR\]\s*', r'logger.error(\1', content)
    content = re.sub(r'logger\.error\((f?["\'])Error:\s*', r'logger.error(\1', content)
    content = re.sub(r'logger\.warning\((f?["\'])\[WARNING\]\s*', r'logger.warning(\1', content)
    content = re.sub(r'logger\.warning\((f?["\'])Warning:\s*', r'logger.warning(\1', content)
    content = re.sub(r'logger\.info\((f?["\'])\[INFO\]\s*', r'logger.info(\1', content)
    content = re.sub(r'logger\.info\((f?["\'])\[OK\]\s*', r'logger.info(\1', content)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

print("Done refactoring logs.")
