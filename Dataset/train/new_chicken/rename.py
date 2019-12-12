import os
dir = os.listdir('./')
i = 60
for file in dir:
	if file != "rename.py":
		os.rename(file,"chicken_"+str(i)+".jpg")
		i+=1
