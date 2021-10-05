import os
import shutil



path="/home/mteg-vas/nisan/workspace/waegan/tmp/user_jpg"
destination="/home/mteg-vas/nisan/workspace/waegan/tmp/results"


files=os.listdir(path)

for file in files:
    folder=(file.split("_")[-1]).split(".")[0]
    print(folder)
    if not os.path.isdir(destination+"/"+folder):
        os.mkdir(destination+"/"+folder)

    shutil.copy(path+"/"+file,destination+"/"+folder+"/"+file)