import sys
import traceback
try:
    import tensorflow as tf
    print("TF version:", tf.__version__)
except Exception as e:
    print("Exception occurred:")
    traceback.print_exc()
