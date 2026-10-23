"""ECG heartbeat arrhythmia classification on the MIT-BIH Arrhythmia Database.

Pipeline: download records -> band-pass filter -> beat extraction -> AAMI
mapping -> inter-patient split -> baseline (RR intervals + Random Forest)
and a 1D CNN -> honest evaluation on unseen patients.
"""

__version__ = "0.1.0"
