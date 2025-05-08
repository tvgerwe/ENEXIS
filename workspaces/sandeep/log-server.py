from flask import Flask, render_template_string

app = Flask(__name__)

LOG_FILE = '../sandeep/ned/logs.txt'

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Logs</title>
    <meta http-equiv="refresh" content="3">
    <style>
        body { 
            font-family: monospace; 
            white-space: pre-wrap; 
            background: #ffffff;  /* White background */
            color: #000000;       /* Black text */
            padding: 20px;
        }
        h2 {
            color: #333333;       /* Dark gray header */
        }
    </style>
</head>
<body>
    <h2>Live Logs</h2>
    <div>{{ logs }}</div>
</body>
</html>
"""

@app.route('/')
def show_logs():
    try:
        with open(LOG_FILE, 'r') as f:
            logs = f.read()
    except FileNotFoundError:
        logs = "Log file not found."
    return render_template_string(HTML_TEMPLATE, logs=logs)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
