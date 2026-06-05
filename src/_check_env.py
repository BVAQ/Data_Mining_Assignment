import sys
print("Python:", sys.executable)
try:
    import sklearn; print("sklearn", sklearn.__version__)
except: print("sklearn: NOT FOUND")
try:
    import xgboost; print("xgboost", xgboost.__version__)
except: print("xgboost: NOT FOUND")
try:
    import lightgbm; print("lightgbm", lightgbm.__version__)
except: print("lightgbm: NOT FOUND")
try:
    import sentence_transformers; print("sentence_transformers", sentence_transformers.__version__)
except: print("sentence_transformers: NOT FOUND")
try:
    import pandas; print("pandas", pandas.__version__)
except: print("pandas: NOT FOUND")
try:
    import numpy; print("numpy", numpy.__version__)
except: print("numpy: NOT FOUND")
try:
    import scipy; print("scipy", scipy.__version__)
except: print("scipy: NOT FOUND")
try:
    import requests; print("requests", requests.__version__)
except: print("requests: NOT FOUND")
