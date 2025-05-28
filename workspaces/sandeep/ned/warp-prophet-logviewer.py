# to run python warp-prophet-logviewer.py
# env - enexis-may-03-env-run

from flask import Flask, render_template_string, Response
from watchfiles import watch
import threading
import time
import os
import re

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "warp-prophet-model-json.log")

app = Flask(__name__)

log_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Log Viewer</title>
    <meta http-equiv="refresh" content="2">
    <style>
        body { font-family: monospace; background: #f7f7fa; color: #000; }
        pre { white-space: pre-wrap; }
        .info { color: #0074D9; }
        .error { color: #FF4136; font-weight: bold; }
        .warning { color: #FF851B; }
        .debug { color: #2ECC40; }
        .logline { margin-bottom: 4px; display: block; }
    </style>
</head>
<body>
    <h2>Live Log: {{ log_path }}</h2>
    <pre>{{ log_content|safe }}</pre>
</body>
</html>
"""

@app.route('/')
def show_log():
    log_content = "(Log file not found)"
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            try:
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                seek_pos = max(0, file_size - 10000)
                f.seek(seek_pos)
                log_content = f.read()
                # Reverse the log lines so latest is at the top
                lines = log_content.splitlines()
                lines = list(reversed(lines))
                # Colorize log levels
                def colorize(line):
                    if 'ERROR' in line:
                        return f'<span class="logline error">{line}</span>'
                    elif 'WARNING' in line:
                        return f'<span class="logline warning">{line}</span>'
                    elif 'INFO' in line:
                        return f'<span class="logline info">{line}</span>'
                    elif 'DEBUG' in line:
                        return f'<span class="logline debug">{line}</span>'
                    else:
                        return f'<span class="logline">{line}</span>'
                log_content = '\n'.join([colorize(l) for l in lines])
            except Exception as e:
                log_content = f"(Error reading log: {e})"
    return render_template_string(log_template, log_path=LOG_PATH, log_content=log_content)

if __name__ == '__main__':
    app.run(debug=True, port=5000)