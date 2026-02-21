# Kiro IDE on Windows → DGX Spark: Complete Setup Guide

## Overview

This guide covers how to run **Kiro IDE on Windows** with your **NVIDIA DGX Spark (192.168.1.211)** as the remote compute target, with the terminal connected to the Spark and project folders synced between your Windows machine and the Spark.

> **Important Note:** As of early 2026, Kiro's native Remote SSH support (via the "Open Remote - SSH" extension) is still maturing and has known issues (authentication flow, version mismatches). This guide provides **two approaches** — try Approach A first, fall back to Approach B if needed.

---

## Prerequisites

| Component | Details |
|-----------|---------|
| **Windows PC** | Windows 10/11 (64-bit) |
| **DGX Spark** | IP: `192.168.1.211`, SSH enabled |
| **Kiro IDE** | Downloaded from [kiro.dev/downloads](https://kiro.dev/downloads/) |
| **NVIDIA Sync** | Downloaded from [build.nvidia.com/spark](https://build.nvidia.com/spark/connect-to-your-spark/sync) |

---

## Step 1: Set Up NVIDIA Sync on Windows

NVIDIA Sync is NVIDIA's official tool for connecting your Windows PC to DGX Spark. It handles SSH key setup and application configuration automatically.

### Install & Connect

1. Download the Windows installer from [build.nvidia.com/spark/connect-to-your-spark/sync](https://build.nvidia.com/spark/connect-to-your-spark/sync)
2. Run the installer
3. On first launch, enter your DGX Spark details:
   - **Hostname or IP**: `192.168.1.211`
   - **Username**: Your DGX Spark user account
   - **Password**: Used only once to set up SSH key-based auth
4. Click **Add** — NVIDIA Sync will configure SSH keys automatically

### Verify SSH Access

After setup, NVIDIA Sync creates an SSH alias. Verify it works:

```powershell
# Open PowerShell or Windows Terminal
ssh abidamera@192.168.1.211
# You should connect without a password prompt
```

Check the SSH config entry NVIDIA Sync creates (you'll need this for Kiro):

```
# Located at C:\Users\damer\.ssh\config
Include "C:\Users\damer\AppData\Local\NVIDIA Corporation\Sync\config\ssh_config"
```

This includes NVIDIA Sync's managed config which already contains:

```
Host 192.168.1.211
    Hostname 192.168.1.211
    User abidamera
    Port 22
    IdentityFile "C:\Users\damer\AppData\Local\NVIDIA Corporation\Sync\config\nvsync.key"
```

Do **not** modify or replace the Include line — NVIDIA Sync manages the SSH key and config automatically.

---

## Step 2: Install Kiro IDE on Windows

1. Download from [kiro.dev/downloads](https://kiro.dev/downloads/)
2. Run the `.exe` installer (right-click → "Run as administrator" if needed)
3. On first launch:
   - Sign in with Google, GitHub, or AWS Builder ID
   - Optionally import VS Code settings
   - Choose your theme
   - Enable the Kiro shell integration

---

## Approach A: Kiro IDE + Open Remote SSH Extension (Try First)

This is the closest to the VS Code Remote SSH experience. Kiro is built on Code OSS, so it supports the Open Remote SSH extension from the Open VSX registry.

### Install the Extension

1. Open Kiro IDE
2. Go to Extensions (`Ctrl+Shift+X`)
3. Search for **"Open Remote - SSH"** by `jeanp413`
4. Install it

### Configure SSH Connection

1. Press `Ctrl+Shift+P` → **"Remote-SSH: Open SSH Configuration File"**
2. Your SSH config already has the NVIDIA Sync Include — no changes needed:

```
Include "C:\Users\damer\AppData\Local\NVIDIA Corporation\Sync\config\ssh_config"
```

   NVIDIA Sync's included config already handles the host, user, and key for `192.168.1.211`.

3. Press `Ctrl+Shift+P` → **"Remote-SSH: Connect to Host"**
4. Enter `192.168.1.211` (or select it if listed)

### Known Issues & Workarounds

- **OAuth/Login stuck**: The authentication callback may not reach the remote Kiro server. Workaround — port-forward the OAuth callback port manually:
  ```powershell
  ssh -L 11106:localhost:11106 abidamera@192.168.1.211
  ```
  Then retry the connection in Kiro.

- **Version mismatch errors**: Clear the remote server cache:
  ```bash
  # SSH into Spark first
  ssh abidamera@192.168.1.211
  rm -rf ~/.kiro-server
  ```
  Then reconnect from Kiro.

- **If Remote SSH fails entirely**, proceed to Approach B below.

### What Works When Connected

- Terminal in Kiro runs directly on DGX Spark (192.168.1.211)
- File explorer shows DGX Spark filesystem
- Code execution happens on DGX Spark (GPU access)
- Kiro's AI features (specs, hooks, steering) work on remote files
- Folders are automatically on the Spark — no separate sync needed

---

## Approach B: Local Kiro IDE + Folder Sync + SSH Terminal (Most Reliable)

If Remote SSH is unreliable, this approach gives you a **local Kiro IDE** with **real-time bidirectional folder sync** to DGX Spark, and an **SSH terminal** connected to the Spark.

### Part 1: Set Up Mutagen for Folder Sync

[Mutagen](https://mutagen.io/) provides fast, real-time, bidirectional file synchronization over SSH.

#### Install Mutagen on Windows

```powershell
# Using winget
winget install Mutagen.Mutagen

# Or download from https://mutagen.io/documentation/introduction/installation
```

#### Create a Sync Session

```powershell
# Sync your local project folder to DGX Spark
mutagen sync create ^
    F:\websites\stockanalysis ^
    abidamera@192.168.1.211:~/stockanalysis ^
    --name=stockanalysis ^
    --sync-mode=two-way-resolved ^
    --ignore=".git,node_modules,__pycache__,.venv,*.pyc"

# Check sync status
mutagen sync list

# Monitor sync activity
mutagen sync monitor stockanalysis
```

#### Manage Sync Sessions

```powershell
# Pause sync
mutagen sync pause stockanalysis

# Resume sync
mutagen sync resume stockanalysis

# Remove sync session
mutagen sync terminate stockanalysis

# Reset if sync gets stuck
mutagen sync reset stockanalysis
```

#### Alternative: rsync via Git Bash or WSL

If you prefer a simpler one-directional push:

```bash
# Push local changes to Spark (from Git Bash or WSL)
rsync -avz --exclude='.git' --exclude='node_modules' --exclude='__pycache__' \
    /f/websites/stockanalysis/ \
    abidamera@192.168.1.211:~/stockanalysis/
```

### Part 2: Configure Kiro Terminal to Auto-SSH into DGX Spark

1. Open Kiro IDE
2. Open your local project folder (the one being synced)
3. Open Settings: `Ctrl+,` → search for **"terminal profile"**
4. Add a custom terminal profile that auto-connects to the Spark:

```json
{
  "terminal.integrated.profiles.windows": {
    "DGX Spark SSH": {
      "path": "ssh",
      "args": ["-t", "abidamera@192.168.1.211", "cd ~/stockanalysis && bash -l"],
      "icon": "server"
    },
    "PowerShell": {
      "source": "PowerShell",
      "icon": "terminal-powershell"
    }
  },
  "terminal.integrated.defaultProfile.windows": "DGX Spark SSH"
}
```

Now every new terminal (`Ctrl+`` `) in Kiro will automatically SSH into your DGX Spark at `192.168.1.211` and land in your project directory.

### Part 3: Optional — SSHFS Mount (Browse Spark Files in Kiro)

If you want Kiro's file explorer to show the DGX Spark filesystem directly:

1. Install [WinFsp](https://winfsp.dev/) and [SSHFS-Win](https://github.com/winfsp/sshfs-win)
2. Map a network drive in File Explorer:
   ```
   \\sshfs\abidamera@192.168.1.211\home\abidamera\stockanalysis
   ```
3. Open this mapped drive as your project folder in Kiro IDE

> **Note**: SSHFS can be slow for large projects. Mutagen sync is faster for day-to-day development. Use SSHFS if you need to browse the full Spark filesystem.

### Workflow Summary

1. **Edit files locally** in Kiro IDE (full AI features — specs, hooks, steering, agentic chat)
2. **Mutagen syncs changes** in real-time to DGX Spark at `192.168.1.211`
3. **Run/test in terminal** which is SSH'd into DGX Spark (GPU access, Docker, CUDA)
4. **Output files sync back** automatically to your Windows machine

---

## Folder Sync Options Comparison

| Tool | Direction | Speed | Setup Complexity |
|------|-----------|-------|-----------------|
| **Mutagen** | Bidirectional, real-time | Fast | Medium |
| **rsync** (manual push) | One-way | Fast | Low |
| **SSHFS/WinFsp** | Mount remote as local drive | Varies (can be slow) | Medium |

---

## Recommended Architecture

```
┌─────────────────────────────┐    SSH + Mutagen Sync    ┌──────────────────────────┐
│   Windows PC                │◄════════════════════════►│   DGX Spark              │
│                             │                          │   192.168.1.211          │
│  ┌─────────────────────┐    │   Bidirectional Sync     │                          │
│  │  Kiro IDE            │    │◄══════════════════════►│  ~/stockanalysis/  │
│  │  - Edit code locally │    │                          │  (GPU compute here)      │
│  │  - Specs & Steering  │    │                          │                          │
│  │  - AI Chat           │    │   SSH Terminal            │  - Python/CUDA runtime   │
│  │  - Agent Hooks       │────│──────────────────────►│  - Docker containers     │
│  └─────────────────────┘    │                          │  - Model training        │
│                             │                          │  - Inference             │
│  NVIDIA Sync (system tray)  │   SSH Key Auth            │                          │
│  - DGX Dashboard access     │◄════════════════════════►│  SSH server (port 22)    │
│  - GPU monitoring           │                          │  Dashboard (port 11000)  │
└─────────────────────────────┘                          └──────────────────────────┘
```

---

## Quick Start Cheat Sheet

```powershell
# ── ONE-TIME SETUP ──

# 1. Verify SSH works
ssh abidamera@192.168.1.211

# 2. Install Mutagen
winget install Mutagen.Mutagen

# 3. Create sync session
mutagen sync create ^
    F:\websites\stockanalysis ^
    abidamera@192.168.1.211:~/stockanalysis ^
    --name=stockanalysis --sync-mode=two-way-resolved ^
    --ignore=".git,node_modules,__pycache__,.venv,*.pyc,.terraform"

# 4. Set Kiro default terminal profile to "DGX Spark SSH" (see Part 2 above)


# ── DAILY WORKFLOW ──

# 1. Open project in Kiro
#    (Kiro opens locally, terminal auto-SSHs to 192.168.1.211)

# 2. Mutagen keeps files in sync automatically
mutagen sync list                   # check status
mutagen sync monitor stockanalysis    # watch sync activity

# 3. Access DGX Dashboard for GPU monitoring
ssh -L 11000:localhost:11000 abidamera@192.168.1.211
# Then open http://localhost:11000 in your browser


# ── CLEANUP ──

# Stop sync when done
mutagen sync pause stockanalysis

# Or terminate sync session entirely
mutagen sync terminate stockanalysis
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| SSH connection refused to `192.168.1.211` | Ensure SSH is enabled on Spark: `sudo systemctl enable --now ssh` |
| SSH asks for password despite key setup | Re-run NVIDIA Sync setup, or manually copy key: `ssh-copy-id abidamera@192.168.1.211` |
| Mutagen sync stuck | Run `mutagen sync reset stockanalysis` |
| Mutagen "permission denied" | Ensure target directory exists on Spark: `ssh abidamera@192.168.1.211 "mkdir -p ~/stockanalysis"` |
| Kiro Remote SSH auth loop | Clear `~/.kiro-server` on Spark, restart Kiro |
| Kiro Remote SSH version mismatch | Update Kiro to latest version, clear `~/.kiro-server` on Spark |
| DGX Dashboard not accessible | Port forward: `ssh -L 11000:localhost:11000 abidamera@192.168.1.211` |
| Slow file operations over SSHFS | Switch to Mutagen for better performance |
| Can't ping `192.168.1.211` | Check both machines are on same subnet, check Spark network settings |
| NVIDIA Sync can't find Spark | Enter `192.168.1.211` directly instead of hostname, wait 3-4 min after Spark boot |
