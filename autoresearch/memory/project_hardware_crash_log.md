---
name: Hardware Crash Diagnosis Log
description: Persistent record of BSOD events, WHEA errors, diagnosed root cause, and mitigation strategy. Read this if laptop crashes again.
type: project
---

## Status: SEVERE DEGRADATION (2026-04-19 — 5 BSODs today)

**Hardware is failing. Do NOT treat as software bug.**

## Today's BSODs (2026-04-19)
| Time | Bugcheck | Name |
|------|----------|------|
| 14:45:57 | 0x0000007f | UNEXPECTED_KERNEL_MODE_TRAP |
| 15:54:55 | 0x000001ca | SYNTHETIC_WATCHDOG_TIMEOUT |
| 16:06:46 | 0x0000001e | KMODE_EXCEPTION_NOT_HANDLED |
| 17:08:50 | 0x00000101 | CLOCK_WATCHDOG_TIMEOUT |
| 17:20:30 | 0x00000101 | CLOCK_WATCHDOG_TIMEOUT |

**All are CPU-core bugchecks = hardware instability.** Crashes every ~1 hour.

## Hardware Confirmed
- **CPU: Intel Core i9-14900HX** — known degradation issue (Intel class action 2024)
- RAM: 32 GB (17.5 GB free at crash time — not a memory issue)
- GPU: RTX 4090 Laptop (idle at crash: 39°C, 0% util, 3.92W — not GPU issue)
- Disk: 306 GB free on C: — not storage issue

## Earlier WHEA Evidence (2026-04-15)
Corrected Machine Check on E-core APIC IDs 16, 17, 24, 25:
- Internal parity error (all)
- Translation Lookaside Buffer Error (APIC 24, 25)

## Root Cause
Intel Raptor Lake (13th/14th gen HX) silicon degradation. Intel issued microcode fix (0x12B) in August 2024 and extended warranty to 5 years.

## Software Mitigations Applied (2026-04-19, after 5th crash)

### Via PowerShell (active now):
1. **CPU max frequency capped at 80%** (`powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 80`)
2. **Turbo Boost disabled** (`powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PERFBOOSTMODE 0`)
3. **156 user processes pinned to P-cores** (affinity 0xFFFF = logical 0-15, avoids E-cores 16-31)
4. Verified: `Get-CimInstance Win32_Processor | CurrentClockSpeed=2200 MHz` (no turbo)

### Via runner (`run_autoresearch.py` already has):
- `_pin_to_safe_cores()` — pins Python to 4 P-core threads at import time
- `AUTORESEARCH_N_THREADS=N` env var override
- `AUTORESEARCH_USE_ALL_CORES=1` to disable (not recommended)

## Tools Installed
- **HWiNFO64** — `C:\Program Files\HWiNFO64\HWiNFO64.EXE` — real-time per-core voltage/temp/freq monitoring

## Tools User Must Install Manually (winget blocked 403)
1. **Intel Processor Diagnostic Tool (IPDT)**:
   - https://www.intel.com/content/www/us/en/download/15951/intel-processor-diagnostic-tool.html
   - Click download, accept license, install the MSI
   - Run it → will self-test CPU → **save the log for RMA evidence**
2. **Intel Extreme Tuning Utility (XTU)**:
   - https://www.intel.com/content/www/us/en/download/17881/intel-extreme-tuning-utility-intel-xtu.html
   - Has built-in CPU stress test + voltage/frequency control

## Firmware/BIOS Fixes (user action required)
### High priority:
1. **Update BIOS to latest** — must include Intel microcode 0x12B (the official Raptor Lake fix)
2. **BIOS**: Disable C6/C7 C-states (keep C1E only)
3. **BIOS**: Set Intel SVID Behavior = "Intel Fail Safe" or "Typical Scenario"
4. **BIOS**: Set CPU Lite Load = Mode 5-8 (more voltage stability)
5. **BIOS**: Disable Turbo Boost entirely (confirm software disable)

### Medium priority:
6. Update chipset drivers (Intel Chipset INF, Intel ME/MEI)
7. Clean GPU driver install (DDU in safe mode)

### If all else fails:
8. **RMA the laptop** — Intel extended Raptor Lake warranty to 5 years. Save IPDT log as evidence.

## Post-Crash Recovery Protocol

When a new session starts after a crash:
1. Read `memory/project_autoresearch_checkpoint.md` (champion state)
2. Read THIS file (hardware context)
3. Verify JSONL integrity: `tail -1` should be complete JSON
4. Check `best_model.pt` exists and is <5MB
5. Rerun champion with seed=0 CPU-only — must reproduce deterministically
6. If reproduction fails: weights may be corrupt. Check git / code_versions
7. Continue from next experiment in checkpoint

## Recommended Work Pattern
- **Short compute bursts only** — MLP trains in 15 sec on CPU, LFM2 takes 300s
- **Never leave unattended** — crashes are stochastic
- **Save after every experiment** — JSONL append is atomic, checkpoint MD is safe
- **Avoid GPU for now** — LFM2 defers until hardware fix
- **Monitor HWiNFO64 during training** — watch VCore voltage, if <1.1V under load that's droop

## Last Updated
2026-04-19 17:25 — Post-5th-crash. Champion: Exp41 Residual MLP, composite +5.499, test Sharpe +6.2113 (verified reproducible).
