# Faster HLS

This repository tracks my experiments on reducing Vitis HLS runtime.

So far, the two most effective techniques are:

1. **Using an in-memory file system** for the project workspace.  
2. **Preloading the mimalloc allocator** (with some optional tuning).

## Current Results  


| Configuration | Mean Runtime | Relative Speed |
|---------------|-------------:|---------------:|
| SSD (`/usr/scratch`) | 240 s | 1.00× |
| In-memory file system | 220 s | 0.91× |
| In-memory FS + mimalloc | 180 s | 0.75× |
| In-memory FS + mimalloc (`MIMALLOC_PURGE_DELAY=200`) | **155 s** | **0.64×** |

These results are from 64 runs per configuration, standard deviation < 1 s.

These measurements use a synthetic design that instantiates ~250 unique small functions, placing heavy I/O load on the Vitis HLS frontend, scheduling, binding, and RTL generation steps. Workflows with lighter file-system activity (e.g., a simple Yosys run) will likely see smaller gains from the in-memory file system.

## Next Steps

The next steps are to profile other EDA tools and see if similar techniques can be applied to them.

This includes tools like: Vivado, Quartus, Yosys, nextpnr, VTR, and OpenROAD.
