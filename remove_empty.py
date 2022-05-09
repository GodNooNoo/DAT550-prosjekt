import os


folders = list(os.walk("test_set/real"))[1:]
for folder in folders:
    if not folder[2]:
        os.rmdir(folder[0])
        print("removed:", folder[0])
