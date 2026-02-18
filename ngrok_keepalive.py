#!/usr/bin/env python3
"""Keep ngrok tunnel alive."""
from pyngrok import ngrok, conf
import time

conf.get_default().auth_token = '38KzuZ6nfhwZQtAiuxBphIPVHpi_tPNG1vS8V3U2YXGEzv3g'

print("Starting ngrok...", flush=True)
tunnel = ngrok.connect(7860, 'http')
url = tunnel.public_url
print(f"PUBLIC_URL={url}", flush=True)

with open('/tmp/ngrok_url_final.txt', 'w') as f:
    f.write(url)

print("Tunnel active, keeping alive...", flush=True)

while True:
    time.sleep(30)
