"""
This module is auto-imported by Python at startup (via the 'site' module).
We set environment variables early to prevent Transformers from importing
TensorFlow/Keras, which avoids Keras 3 compatibility errors.
"""

import os

# Hard-disable TensorFlow backend in Transformers
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("DISABLE_TF", "1")
os.environ.setdefault("USE_TF", "0")

# Silence verbose TF logs if it does get pulled in indirectly
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Prefer legacy Keras behavior if TensorFlow is present
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# Reduce oneDNN and deprecation chatter
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


