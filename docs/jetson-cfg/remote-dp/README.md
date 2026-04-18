# 🖥️ Jetson Remote Display Configuration

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