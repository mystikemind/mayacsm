#!/bin/bash
# Start Maya with public URL for live testing
# This script:
# 1. Starts ngrok tunnel in background
# 2. Starts Maya server
# 3. Prints the public URL to access from your browser

cd /home/ec2-user/SageMaker/project_maya

# Kill any existing ngrok/maya processes
pkill -f "ngrok" 2>/dev/null
pkill -f "run.py" 2>/dev/null
sleep 2

echo "=========================================="
echo "  MAYA LIVE TEST"
echo "=========================================="

# Start ngrok in background
echo "Starting ngrok tunnel..."
python3 -c "
from pyngrok import ngrok, conf
conf.get_default().auth_token = '38KzuZ6nfhwZQtAiuxBphIPVHpi_tPNG1vS8V3U2YXGEzv3g'
tunnel = ngrok.connect(8000, 'http')
print('='*50)
print('PUBLIC URL:', tunnel.public_url)
print('='*50)
print('Open this URL in your browser (with microphone)')
print('='*50)
with open('/tmp/maya_public_url.txt', 'w') as f:
    f.write(tunnel.public_url)
import time
while True:
    time.sleep(30)
" &

NGROK_PID=$!
sleep 5

# Show the URL
echo ""
cat /tmp/maya_public_url.txt 2>/dev/null || echo "Waiting for ngrok..."
echo ""

# Start Maya server (this blocks)
echo "Starting Maya server on port 8000..."
echo "Press Ctrl+C to stop"
echo ""
python3 run.py --port 8000

# Cleanup on exit
kill $NGROK_PID 2>/dev/null
