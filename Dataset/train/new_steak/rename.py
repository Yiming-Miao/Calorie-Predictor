import os
dir = os.listdir('./')
i = 82
for file in dir:
	if file != "rename.py":
		os.rename(file,"steak_"+str(i)+".jpg")
		i+=1
