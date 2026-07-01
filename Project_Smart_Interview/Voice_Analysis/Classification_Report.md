# Voice Classification Timeline Report

This report documents the voice state classifications (Hesitant, Stressed, Enthusiastic, Confident) calculated in **10-second intervals** for each speaker.

## Heuristic Definitions & Calibration
The classifier calculates Z-scores relative to the average of this recording session to adjust for differences in speaker biology (gender, vocal pitch range) and microphone setups.

| State | Primary Acoustic Profile |
| :--- | :--- |
| **Confident** | High voice stability (low jitter/shimmer), strong resonance (high HNR), stable loudness. |
| **Enthusiastic** | High pitch, high pitch variability, elevated loudness, high spectral flux (dynamic). |
| **Stressed** | High pitch, elevated loudness, high vocal fold instability (jitter/shimmer), low HNR. |
| **Hesitant** | Low loudness, low pitch, high vocal fold jitter/shimmer (tremulous), flat articulation. |
| **Inactive** | Very low energy/volume levels (silence or non-speech duration). |

## 👤 Speaker001

| Time Interval | Classified State | Dominant Characteristic / Feature Highlights |
| :--- | :--- | :--- |
| **0s - 10s** | **Hesitant** | Clear tone, Soft projection |
| **10s - 20s** | **Hesitant** | Lower Pitch, Clear tone, Soft projection |
| **20s - 30s** | **Enthusiastic** | Elevated Pitch, Vocal instability, Loud projection |

