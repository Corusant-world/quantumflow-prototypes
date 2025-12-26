# SIGMA: MC/DC Static Analysis Report
Source: integrations/sigma_cfs_app/src/sigma_core.c

| Decision | Conditions | Independence Required | Status |
|---|---|---|---|
| `if(!initialized)` | 1 | 0 | N/A (Simple Decision) |
| `if(frame->engine_thrust < NOISE_FLOOR && dv_sq > 0.1f)` | 2 | 2 | PENDING (Test Case Required) |
| `if(frame->frame_id != last_frame.frame_id + 1)` | 1 | 0 | N/A (Simple Decision) |