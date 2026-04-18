
# 🚀 Setup Jetson AP Mode (Auto Start at Boot)

This guide configures the Jetson device to automatically start a WiFi hotspot (AP mode) on boot using nmcli and a startup script. 

> [!NOTE]
> This setup provides a reliable connection between a laptop and the Jetson, enabling the use of remote access tools (e.g., [AnyDesk](https://anydesk.com/en/downloads/linux)) for convenient GUI interaction. It is essential for *on-device debugging* in the network-constrained environments such as *remote* or *field deployments*.

## 1. Check the wireless interface name on your Jetson
```bash
ifconfig
```

Typical interface names look like: `wlP1p1s0`, `wlan0`, etc.


## 2. Create AP Startup Script
```bash
sudo vim /usr/local/bin/start_ap.sh
```

Add the following content:
```bash
#!/bin/bash

# Wait for system and network initialization
sleep 3

# Enable Wi-Fi
nmcli radio wifi on

# Disconnect any existing connection to avoid conflicts
nmcli device disconnect <WIRELESS_INTERFACE> 2>/dev/null

# Start hotspot
nmcli device wifi hotspot ifname <WIRELESS_INTERFACE> ssid <AP_NAME> password <AP_PASSWORD>
```

Replace `<WIRELESS_INTERFACE>`, `<AP_NAME>`, and `<AP_PASSWORD>` with your actual values.

## 3. Make the script executable
```bash
sudo chmod +x /usr/local/bin/start_ap.sh
```

## 4. Create systemd Service
Create a dedicated service:
```bash
sudo vim /etc/systemd/system/jetson-ap.service
```

Paste the following:
```bash
[Unit]
Description=Jetson WiFi Access Point
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/start_ap.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## 5. Enable and Start Service
Reload systemd and enable the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable jetson-ap.service
sudo systemctl start jetson-ap.service
```

## 6. Debugging (Optional)
Check service status:
```bash
systemctl status jetson-ap.service
```

View the logs:
```bash
journalctl -u jetson-ap.service -f
```
