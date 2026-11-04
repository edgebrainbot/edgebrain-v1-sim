# Jetson-RT-Kernel

This document contains the instruction for building a **real-time (RT)** kernel for [Nvidia Jetson AGX Orin (Developer-Kit)](https://developer.nvidia.com/embedded/learn/jetson-agx-orin-devkit-user-guide/index.html) along with an installation script. A precompiled kernel is built and provided with the following system versions:

* *JetPack Version - 6.2*
* *NVIDIA Jetson Linux - 36.4.3*
* *Linux kernel Version - 5.15.148*


> [!NOTE] 
> The Real-Time kernel introduces preemptive scheduling, significantly improving system responsiveness for real-time workloads. We `strongly recommend` applying the RT patch for latency-sensitive applications.


> [!WARNING] 
> Although NVIDIA provides an [OTA-based](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/Kernel/KernelCustomization.html#real-time-kernel-using-ota-update) RT kernel installation, it has been reported to cause [boot-time freezes](https://forums.developer.nvidia.com/t/boot-freezing-when-installing-preemptrt-on-nvme-setup-with-agx-orin-dev-kit-jetpack-6-2/323869) when entering the GUI. Manual installation is recommended for stability.

## Kernel Compile

If you’re new to Jetson development, we suggest reading [Jetson Developer Guide](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/index.html) beforehand. To build the RT kernel manually, refer to NVIDIA’s official instructions: [Kernel Customization Guide](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/Kernel/KernelCustomization.html#sd-kernel-kernelcustomization). Following this guide will produce:

   * A bindary real-time kernel image
   * The kernel module 5.15.148-rt-tegra
   * Device Tree Blobs (DTBs) for boot configuration

<!-- > **Note:** Ensure your Jetson system matches the required L4T version to avoid compatibility issues. -->

> [!TIP]
> (Optional) If UEFI Secure Boot is `disabled`, you can skip the kernel signing and encryption steps. Otherwise, follow Nvidia's [Secure Boot Guide](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/Security/SecureBoot.html#sd-security-secureboot).

## Installation

There're two options provided to install the compiled kernel:
   * Flash using `flash.sh`: ideal for full image flashing. See [flashing support](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/FlashingSupport.html) for more details.
   * In-place installation by `install_rt_kernel.sh`: apply the precompiled RT kernel directly on the device.

For the second option, first download the [precompiled image](https://www.dropbox.com/scl/fi/71iaga4o9ycmgp9tlll9a/compiled.zip?rlkey=z8a0ku992ypt0cbv4jibmeuqu&st=bofxgpuk&dl=0) and unpack into this directory. For safety consideration, backup your current boot configuration:

```bash
sudo cp -rf /boot /boot_bak
```

Install the RT kernel by specifying the storage type (e.g., `nvme` or `emmc`):

```bash
cd Jetson-RT-Kernel
sudo ./install_rt_kernel.sh --storage <nvme|emmc>
```

Running this script will: 

   * Install DTBs to `/boot`
   * Decompress kernel modules to `/lib/modules`
   * Install the RT kernel image `Image.rt` to `/boot`
   * Install the RT initrd `initrd.img-5.15.148-rt-tegra` to `/boot`
   * Add a new boot entry to `/boot/extlinux/extlinux.conf`

Once complete, reboot and select the real-time kernel from the boot menu.

> [!IMPORTANT]
> To make RT kernel the default boot option, set `DEFAULT real-time` in `/boot/extlinux/extlinux.conf`

<!-- 
Here are some useful links for reference: 
- https://chipnbits.github.io/content/projects/RLUnicycle/rtkernel/rtpatch.html
- https://forums.developer.nvidia.com/t/preempt-rt-patches-for-jetson-nano/72941
- https://forums.developer.nvidia.com/t/jetson-agx-orin-rt-linux-without-reflashing/283832
- https://forums.developer.nvidia.com/t/no-display-with-preempt-rt-patches/240876
- https://forums.developer.nvidia.com/t/build-the-real-time-kernel/229571
- https://blog.csdn.net/weixin_43854380/article/details/126584835
- https://github.com/kozyilmaz/nvidia-jetson-rt/blob/master/docs/README.03-realtime.md -->


## Configuration and Test

### Verifying RT Kernel Installation

After rebooting, confirm that the RT kernel is loaded:

```bash
uname -vr
```

The expected output should be:

```bash
5.15.148-rt-tegra #1 SMP PREEMPT_RT
```


### Real time test using latency plot under the stress

Go to the `/test` file and Run the RT test

   ```bash
   cd test
   sudo ./rt-test.sh --cores <number-of-cores>
   ```

Dependencies (`rt-tests`, `stress`, `gnuplot`) will be installed automatically. If `<number-of-cores>` are not specified, the script  will default to the number of available CPU cores (`nproc`).

## Testing Result

Below are example latency plots on the Jetson AGX Orin using 4 and 8 CPU cores:


| 4 Cores (non real-time kernel)                                        | 4 Cores (real-time kernel)                                        |
|-----------------------------------------------------------------------|-------------------------------------------------------------------|
| <img src="./test/results/4-cores/non-rt.png" height="300" alt="rlm"/> | <img src="./test/results/4-cores/rt.png" height="300" alt="rlm"/> |

| 8 Cores (non real-time kernel)                                        | 8 Cores (real-time kernel)                                        |
|-----------------------------------------------------------------------|-------------------------------------------------------------------|
| <img src="./test/results/8-cores/non-rt.png" height="300" alt="rlm"/> | <img src="./test/results/8-cores/rt.png" height="300" alt="rlm"/> |


Analysis:
- More sample on the left means lower latency in general
- More clustered samples indicate less flutter
- The max latency should not deviate far from mean value
