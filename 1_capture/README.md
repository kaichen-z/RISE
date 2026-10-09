# 1 — Data capture (Lua, TI mmWave Studio)

Scripts for the TI 4-chip cascade board (MMWCAS-RF-EVM + MMWCAS-DSP-EVM), run from the Lua
console of mmWave Studio on the capture PC.

| file | purpose |
|------|---------|
| `Cascade_Configuration_MIMO.lua` | chirp / MIMO profile configuration (v1) |
| `Cascade_Configuration_MIMO_v2.lua` | configuration used for the released captures |
| `Cascade_Capture.lua` | arm the DCA and stream frames to disk (v1) |
| `Cascade_Capture_v2.lua` | capture loop used for the released captures |

Order: load the configuration script first, then the capture script. Output is a set of
`*.bin` files plus the `.mmwave.json` / `.setup.json` produced by mmWave Studio — that whole
folder is the input of stage 2.
