import os
dir = os.listdir('./')
i = 0
for file in dir:
	if file != "rename.py":
		os.rename(file,"pizza_"+str(i)+".jpg")
		i+=1
