import shutil
import os

path = '/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/config'
if os.path.exists(path):
    print(f"Removing {path}...")
    try:
        shutil.rmtree(path)
        print("Success.")
    except Exception as e:
        print(f"Error: {e}")
else:
    print("Path does not exist.")
