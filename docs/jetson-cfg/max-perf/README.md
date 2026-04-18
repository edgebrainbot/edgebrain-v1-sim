# Jetson Configuration

This document contains the instruction for building a **real-time (RT)** kernel for [Nvidia Jetson AGX Orin (Developer-Kit)](https://developer.nvidia.com/embedded/learn/jetson-agx-orin-devkit-user-guide/index.html) along with an installation script. A precompiled kernel is built and provided with the following system versions:

* *JetPack Version - 6.2*
* *NVIDIA Jetson Linux - 36.4.3*
* *Linux kernel Version - 5.15.148*


## 🚀 Enable MAX Performance Mode on Jetson (Boot-Time)

This guide explains how to automatically enable **MAXN performance mode** on Jetson at boot using a systemd service.

Applies to:

- Jetson AGX Orin  
- Jetson Orin NX  
- Jetson Xavier Series  

---

### 📌 What This Does

At boot, the system will automatically execute:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```
This will:

- Set power mode to MAXN

- Lock CPU/GPU/EMC to maximum frequency

- Prevent runtime downclocking

The overall steps:

1. Create the systemd Service File
```bash
sudo vim /etc/systemd/system/jetson-maxperf.service
```

2. Add the Following Content
```bash
[Unit]
Description=Jetson Max Performance Mode
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/sbin/nvpmodel -m 0
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=true

[Install]
WantedBy=multi-user.target
```

3. Reload systemd
```bash
sudo systemctl daemon-reload
```

4. Enable at Boot
```bash
sudo systemctl enable jetson-maxperf.service
```

▶ Optional: Run Immediately
```bash
sudo systemctl start jetson-maxperf.service
```

🔍 Verify It Is Working
```bash
sudo systemctl status jetson-maxperf.service
```


## 🚀 Set Jetson AP Mode (Auto Start at Boot)

This guide configures the Jetson device to automatically start a WiFi hotspot (AP mode) on boot using nmcli and a lightweight startup script.

1. Create AP Startup Script
```bash
sudo vim /usr/local/bin/start_ap.sh
```

Paste the following content:
```bash
#!/bin/bash

# Wait for system and network initialization
sleep 8

# Enable WiFi
nmcli radio wifi on

# Disconnect any existing connection to avoid conflicts
nmcli device disconnect wlP1p1s0 2>/dev/null

# Start hotspot
nmcli device wifi hotspot ifname wlP1p1s0 ssid JetsonAP password edge1234
```

Make the script executable:
```bash
sudo chmod +x /usr/local/bin/start_ap.sh
```

2. Create rc-local Service
Create a systemd service to enable /etc/rc.local support:
```bash
sudo vim /etc/systemd/system/rc-local.service
```

Add the following:
```bash
[Unit]
Description=/etc/rc.local Compatibility
ConditionPathExists=/etc/rc.local

[Service]
Type=forking
ExecStart=/etc/rc.local start
TimeoutSec=0
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

3. Create rc.local Script
Create the rc.local entry point:
```bash
sudo nano /etc/rc.local
```
Add:
```bash
#!/bin/bash

/usr/local/bin/start_ap.sh &

exit 0
```

Make it executable:
```bash
sudo chmod +x /etc/rc.local
```

4. Enable and Start Service
Reload systemd and enable the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable rc-local.service
sudo systemctl start rc-local.service
```


For remote onboard Jetson debug, you can use other remote connection tool (e.g., *Anydesk*, *Teamviewer*). However, these softwares are not typically useful if the Jetson is not connected to a monitor. According to the [solution](https://askubuntu.com/questions/453109/add-fake-display-when-no-monitor-is-plugged-in), there're two options:

1. purchase a dummy display plug [DP Port](https://www.amazon.com/FUERAN-DP-DisplayPort-emulator-2560x1600/dp/B075PTQ4NH/ref=sr_1_2_sspa?dib=eyJ2IjoiMSJ9._2VBeAusIsfeL9L3oHOyN6PCekAraiSOV3Mbqo9epsJwtAFnpeQKXHkz0wJTW8nPMM9W9X-Z_Sbt1gaplwMw2RqV9BnFry5G4bcKFI4PVCb8FD_8RxNh8B5D97RcYCU7aVqnbDOkPJZwz5UVYRL-jNVwVQtxFcuWtiiWVWm0XvcXtdtGp9HA9WySLN1MFOEmpgGPhkNvBDvJuxqC3YC28CEOz7CkhmC0gsXiwtJNmkA.K023GTNCJhfrzl89n0sznwDJ5qqfZ3KOK0XIoN7h2XE&dib_tag=se&keywords=dp%2Bport%2Bdummy%2Bport&qid=1750013547&sr=8-2-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1)
2. turn the video output to a 
```bash
sudo apt-get install xserver-xorg-video-dummy
```

Then add the content in `/etc/X11/xorg.conf`:
```bash
Section "Device"
    Identifier  "Configured Video Device"
    Driver      "dummy"
    # Default is 4MiB, this sets it to 16MiB
    VideoRam    16384
EndSection

Section "Monitor"
    Identifier  "Configured Monitor"
    HorizSync 31.5-48.5
    VertRefresh 50-70
EndSection

Section "Screen"
    Identifier  "Default Screen"
    Monitor     "Configured Monitor"
    Device      "Configured Video Device"
    DefaultDepth 24
    SubSection "Display"
    Depth 24
    Modes "1024x800"
    EndSubSection
EndSection
```