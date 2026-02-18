#!/usr/bin/env python3
"""Keep ngrok tunnel alive."""
from pyngrok import ngrok, conf
import time

conf.get_default().auth_token = '38KzuZ6nfhwZQtAiuxBphIPVHpi_tPNG1vS8V3U2YXGEzv3g'

tunnel = ngrok.connect(7860, 'http')
url = tunnel.public_url

with open('/tmp/ngrok_public_url.txt', 'w') as f:
    f.write(url)

print(f"PUBLIC URL: {url}", flush=True)

# Keep alive
while True:
    time.sleep(30)
