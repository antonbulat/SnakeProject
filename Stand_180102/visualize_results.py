import matplotlib.pyplot as plt
import numpy as np

mystr="0;466;553;893;1278;819;535;402;262;155;100;74;47;33;20;13;12;4;4;2;2;0;1;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0"
randomm=mystr.split(";")
pltrandom=[]
for i in range(0,len(randomm)):
    for j in range(0,int(randomm[i])):
        pltrandom.append(i)

apple_per_game=sum([int(randomm[i])*i for i in range(0,len(randomm))])/sum([int(randomm[i])for i in range(0,len(randomm))])
plt.title("2 h, increase in size by 100%, apples per game = "+str(apple_per_game))
plt.xticks(range(0,30))
plt.hist(pltrandom,np.arange(0, 30, 1),color='g')
plt.show()