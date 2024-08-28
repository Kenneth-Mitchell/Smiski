import os

for file in os.listdir("smiski_images"):
    #get file name
    name = file.split(".")[0]
    #make dir with file name
    os.makedirs(f"siamese_data/{name}", exist_ok=True)
    #copy file to new dir
    os.system(f"cp smiski_images/{file} siamese_data/{name}/{file}")