# 🚀 Enable MAX Performance Mode on Jetson (Boot-Time)

This guide explains how to automatically enable **MAXN performance mode** on Jetson at boot using a systemd service, which applies to:

- Jetson AGX Orin  
- Jetson Orin NX  
- Jetson Xavier Series  

## 📌 What This Does

At boot, the system will automatically execute:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```
This will:

- Set power mode to MAXN

- Lock CPU/GPU/EMC to maximum frequency

- Prevent runtime downclocking

## 1. Create the systemd Service File
```bash
sudo vim /etc/systemd/system/jetson-maxperf.service
```

## 2. Add the Following Content
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

## 3. Reload systemd
```bash
sudo systemctl daemon-reload
```

## 4. Enable at Boot
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


