# HPC Jobs for NTNU IDUN

A collection of SLURM job scripts, utility tools, and log-management conventions for running JPDVT experiments on the NTNU HPC cluster.

---

## 📁 Directory Structure

```
.
├── README.md
├── utils/
│   └── check_storage.sh       # disk‐usage checker and cleanup helper
├── slurm/
│   ├── infer_JPDVT.slurm      # single‐GPU inference
│   ├── infer_JPDVT_4x4.slurm  # 4×4‐card inference
│   ├── train_JPDVT.slurm      # single‐GPU training
│   └── train_JPDVT4x4.slurm   # 4×4‐card training
└── logs/
    ├── out/                   # all “.out” logs
    ├── err/                   # all current “.err” logs (ignored by Git; contains `.gitkeep`)
    └── err.old/               # archived “.err.old” logs
```

---

## 🚀 Getting Started

1. **Clone the repo**  
   ```bash
   git clone https://github.com/hamzafer/hpc-jobs-ntnu.git
   cd hpc-jobs-ntnu
   ```

2. **Inspect available scripts**  
   - **SLURM jobs** in `slurm/`  
   - **Utility** in `utils/check_storage.sh`  

3. **Ensure executables**  
   ```bash
   chmod +x utils/check_storage.sh
   ```

---

## ⚙️ Usage

### 1. Submit a SLURM job

```bash
cd slurm
sbatch train_JPDVT.slurm
# or for 4×4 runs:
sbatch train_JPDVT4x4.slurm
```

Modify each script’s SBATCH header (partition, GPUs, time, etc.) to suit your project.

### 2. Check storage & cleanup

Use the utility script to monitor your `$TMPDIR` or scratch space:

```bash
./utils/check_storage.sh /path/to/scratch
```

It will report disk usage and optionally remove old output/log files.

---

## 🗄️ Log Management

- **`.out` files** → `logs/out/`  
- **Current `.err` files** → `logs/err/` (tracked only as an empty folder via `.gitkeep`)  
- **Old `.err.old` files** → `logs/err.old/`

> **Note:** All `*.err` files are listed in `.gitignore`, so they remain on your local machine but never get committed to `main`. To archive them remotely, check out the `logs-archive` branch.

---

 © 2025 Hamza
