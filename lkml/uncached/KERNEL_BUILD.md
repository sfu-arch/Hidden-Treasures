# Building Linux Kernel from Source with CMA Support

This guide walks you through building a custom Linux kernel with CMA (Contiguous Memory Allocator) enabled for your CMA cache module testing.

## Prerequisites

### 1. Install Build Dependencies
```bash
sudo apt update
sudo apt install -y \
    build-essential \
    libncurses-dev \
    bison \
    flex \
    libssl-dev \
    libelf-dev \
    bc \
    rsync \
    kmod \
    cpio \
    initramfs-tools
```

### 2. Free Up Disk Space
Building a kernel requires ~15-20GB of free space:
```bash
# Check available space
df -h /

# Clean up if needed
sudo apt autoremove
sudo apt autoclean
```

## Step-by-Step Kernel Build

### 1. Download Kernel Source
```bash
# Create build directory
mkdir -p ~/kernel-build
cd ~/kernel-build

# Download stable kernel (adjust version as needed)
wget https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.1.55.tar.xz

# Or download latest stable
# wget https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.6.3.tar.xz

# Extract
tar -xf linux-6.1.55.tar.xz
cd linux-6.1.55
```

#### Option C: Use Ubuntu Kernel Source
```bash
# Install build dependencies for the running Ubuntu kernel
sudo apt update
# Enable source repositories if not already enabled
sudo sed -i 's/^# deb-src /deb-src /' /etc/apt/sources.list
sudo apt update
sudo apt-get build-dep linux
sudo apt source linux-image-$(uname -r)
# Download Ubuntu kernel source package
apt source linux-image-$(uname -r)
# Enter the source directory (may vary by version)
cd linux-*

# Install Debian packaging tools
sudo apt install -y devscripts fakeroot

# Build Debian kernel packages (signed HWE variants included)
dpkg-buildpackage -uc -us -b -j$(nproc)

# Install generated kernel and headers
cd ..
sudo dpkg -i linux-image-*_*.deb linux-headers-*_*.deb
```

### 2. Configure Kernel

#### Option A: Start with Current Config (Recommended)
```bash
# Copy current running kernel config
cp /boot/config-$(uname -r) .config

# Update config for new kernel version
make olddefconfig
```

#### Option B: Default Config
```bash
# Use default config for your architecture
make defconfig
```

### 3. Enable CMA and Related Options
```bash
# Open kernel configuration menu
make menuconfig
```

**Navigate and enable these options:**

**Memory Management Options:**
```
Device Drivers →
  [*] DMA Engine support →
    [*] DMA Engine debugging
    [*] DMA Engine verbose debugging

Memory Management options →
  [*] Contiguous Memory Allocator (CMA)
  [*] CMA debug messages (DEVELOPMENT)
  (16) Default contiguous memory area size in megabytes
  [*] CMA debugfs interface
  [*] Enable CMA areas map
```

**DMA Options:**
```
Device Drivers →
  Generic Driver Options →
    [*] DMA Contiguous Memory Allocator
    [*] Selected region size alignment
    (0) Maximum PAGE_SIZE order of alignment for DMA IOMMU buffers
```

**Platform Device Support:**
```
Device Drivers →
  [*] Platform device drivers
```

### 4. Configure CMA-Specific Settings

#### Option A: Via menuconfig (GUI)
In menuconfig, set:
- `CONFIG_CMA=y`
- `CONFIG_CMA_DEBUG=y` 
- `CONFIG_CMA_DEBUGFS=y`
- `CONFIG_CMA_AREAS=7`
- `CONFIG_CMA_SIZE_MBYTES=64` (or higher)

#### Option B: Direct config edit
```bash
# Add/modify these lines in .config
echo "CONFIG_CMA=y" >> .config
echo "CONFIG_CMA_DEBUG=y" >> .config
echo "CONFIG_CMA_DEBUGFS=y" >> .config
echo "CONFIG_CMA_AREAS=7" >> .config
echo "CONFIG_CMA_SIZE_MBYTES=64" >> .config
echo "CONFIG_DMA_CMA=y" >> .config

# Resolve dependencies
make olddefconfig
```

### 5. Verify CMA Configuration
```bash
# Check that CMA is enabled
grep -E "CONFIG_CMA|CONFIG_DMA_CMA" .config

# Should show:
# CONFIG_CMA=y
# CONFIG_CMA_DEBUG=y
# CONFIG_CMA_DEBUGFS=y
# CONFIG_DMA_CMA=y
```

### 6. Build the Kernel

#### Set Build Parameters
```bash
# Use all CPU cores for faster build
export MAKEFLAGS="-j$(nproc)"

# Estimate build time (usually 30-90 minutes)
echo "Starting build with $(nproc) cores..."
echo "Estimated time: 30-90 minutes depending on your system"
```

#### Start Build
```bash
# Build kernel, modules, and device tree blobs
time make -j$(nproc)

# Build modules
time make modules -j$(nproc)
```

### 7. Install the Kernel

#### Simple Approach: Use LOCALVERSION to Keep Both Kernels

The easiest way to keep your current kernel intact is to give your new kernel a unique version suffix:

```bash
# Set a custom local version before building
export LOCALVERSION=-custom-cma
# Or edit .config directly
echo "CONFIG_LOCALVERSION=\"-custom-cma\"" >> .config
make olddefconfig

# Now build and install - this creates a separate kernel
make -j$(nproc)
sudo make modules_install
sudo make install

# Update GRUB to see both kernels
sudo update-grub
```

This approach:
- Installs your new kernel alongside the existing one
- Creates `/boot/vmlinuz-<version>-custom-cma` (separate from current)
- Creates `/lib/modules/<version>-custom-cma/` (separate module directory)
- Both kernels appear in GRUB menu
- No backup needed - your original kernel stays untouched

#### Alternative: Use Debian Package Method (Even Safer)
```bash
# Build as Debian package with custom version
make -j$(nproc) bindeb-pkg LOCALVERSION=-custom-cma

# Install the packages (keeps existing kernel packages intact)
cd ..
sudo dpkg -i linux-image-*custom-cma*.deb linux-headers-*custom-cma*.deb

# Update GRUB
sudo update-grub
```

#### Manual Backup Method (If You Prefer)
If you still want explicit backups:

```bash
# Get current kernel version
CURRENT_KVER=$(uname -r)

# Create backup directory
sudo mkdir -p /tmp/kernel-backup/{boot,modules}

# Backup current kernel files
sudo cp /boot/vmlinuz-"$CURRENT_KVER" /tmp/kernel-backup/boot/
sudo cp /boot/initrd.img-"$CURRENT_KVER" /tmp/kernel-backup/boot/
sudo cp /boot/System.map-"$CURRENT_KVER" /tmp/kernel-backup/boot/
sudo cp /boot/config-"$CURRENT_KVER" /tmp/kernel-backup/boot/

# Backup current modules
sudo cp -a /lib/modules/"$CURRENT_KVER" /tmp/kernel-backup/modules/

# List what was backed up
echo "Backed up kernel version: $CURRENT_KVER"
ls -la /tmp/kernel-backup/boot/
ls -la /tmp/kernel-backup/modules/
```

#### Install Modules (if not using Debian packages)
```bash
sudo make modules_install
```

#### Install Kernel (if not using Debian packages)
```bash
sudo make install
```

#### Update Initramfs and GRUB
```bash
# Update initramfs for new kernel
sudo update-initramfs -c -k $(make kernelrelease)

# Update GRUB bootloader
sudo update-grub
```

### 8. Configure Boot Parameters

Add CMA boot parameters to GRUB:
```bash
sudo nano /etc/default/grub
```

Modify the line:
```bash
# Before:
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"

# After (choose appropriate CMA size):
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash cma=128M"
```

Update GRUB:
```bash
sudo update-grub
```

### 9. Boot into New Kernel

```bash
sudo reboot
```

**During boot:**
1. **Access GRUB menu** - Hold SHIFT during boot or press ESC/F12
2. **Select Advanced options** for Ubuntu
3. **Choose your new kernel** from the list (should be at the top)
4. **Boot normally**

After successful boot with new kernel:

1. **Select your new kernel** from GRUB menu (if not default)

2. **Verify CMA is working:**
```bash
# Check kernel version
uname -r

# Check CMA in kernel config
zcat /proc/config.gz | grep -E "CONFIG_CMA|CONFIG_DMA_CMA"

# Check CMA memory info
cat /proc/meminfo | grep -i cma

# Check CMA debug info (if enabled)
ls /sys/kernel/debug/cma/
```

3. **Test your module:**
```bash
cd /localhome/ashriram/Hidden-Treasures/lkml/uncached
sudo insmod cma_cache.ko
sudo chmod 666 /dev/cma_cache /sys/kernel/cma_cache/command
./cma_test basic
```

## Recovery and Rollback

### If New Kernel Fails to Boot

1. **Boot into previous kernel:**
   - Restart and select the previous kernel from GRUB menu
   - Your old kernel should still be available in "Advanced options"

2. **Restore from backup if needed:**
   ```bash
   # Restore kernel files
   CURRENT_KVER=$(uname -r)
   sudo cp /tmp/kernel-backup/boot/* /boot/
   
   # Restore modules
   sudo rm -rf /lib/modules/"$CURRENT_KVER"
   sudo cp -a /tmp/kernel-backup/modules/"$CURRENT_KVER" /lib/modules/
   
   # Update initramfs and GRUB
   sudo update-initramfs -u -k "$CURRENT_KVER"
   sudo update-grub
   ```

3. **Remove problematic kernel:**
   ```bash
   # List installed kernels
   dpkg -l | grep linux-image
   
   # Remove problematic kernel (replace with actual version)
   sudo apt remove linux-image-<version>
   sudo update-grub
   ```

### If New Kernel Boots but Modules Don't Work

1. **Check module loading:**
   ```bash
   # Check if modules are loading
   lsmod
   
   # Check for module errors
   dmesg | grep -i error
   dmesg | grep -i module
   ```

2. **Rebuild and reinstall modules:**
   ```bash
   cd ~/kernel-build/linux-*
   make modules_install
   sudo depmod -a
   ```

3. **Restore module backup if needed:**
   ```bash
   CURRENT_KVER=$(uname -r)
   sudo rm -rf /lib/modules/"$CURRENT_KVER"
   sudo cp -a /tmp/kernel-backup/modules/"$CURRENT_KVER" /lib/modules/
   sudo depmod -a
   ```

### Clean Up After Successful Installation

Once you're confident the new kernel works properly:

```bash
# Remove backup
sudo rm -rf /tmp/kernel-backup

# Remove old kernel packages (optional)
sudo apt autoremove
```

## Expected Results

After successful build and boot:

```bash
# Should show CMA memory
$ cat /proc/meminfo | grep -i cma
CmaTotal:     131072 kB
CmaFree:      130048 kB

# CMA test should work with larger allocations
$ ./cma_test basic
=== Basic Allocation Test ===
Allocating 1M...
Successfully allocated 1M with ID 1
Allocating 2M...
Successfully allocated 2M with ID 2
Allocating 4M...
Successfully allocated 4M with ID 3
...
```

## Performance Optimization

### Faster Builds
```bash
# Use ccache for faster rebuilds
sudo apt install ccache
export PATH="/usr/lib/ccache:$PATH"

# Use distcc for distributed builds (if you have multiple machines)
sudo apt install distcc
```

### Kernel Size Optimization
```bash
# Remove unnecessary modules
make localmodconfig  # Only builds modules for currently loaded drivers

# Strip debug symbols to reduce size
scripts/config --disable DEBUG_INFO
```

## Advanced Options

### Custom CMA Configuration

For specific use cases, you can configure multiple CMA regions:

```bash
# In .config
CONFIG_CMA_SIZE_MBYTES=128
CONFIG_CMA_AREAS=7

# Boot parameter for multiple regions
cma=64M@0x00000000 cma=64M@0x40000000
```

### Testing Different Kernel Versions

```bash
# Download and test different versions
wget https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.6.3.tar.xz  # Latest
wget https://cdn.kernel.org/pub/linux/kernel/v5.x/linux-5.15.138.tar.xz  # LTS
```

This comprehensive build process will give you a kernel with full CMA support, allowing your CMA cache module to allocate much larger contiguous memory blocks reliably.
