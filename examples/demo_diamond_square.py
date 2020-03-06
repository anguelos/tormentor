from diamond_square import diamond_square
import timeit
from matplotlib import pyplot as plt

device="cpu"
roughness=1.0
do_show=True

if __name__=="__main__":
    img = diamond_square(N=10)[0,0,:,:]
    plt.imshow(img)
    plt.show()
    pass
    #img=grow_plasma(10,roughness,device=device)
    #show(img,"result")
    #img=(1-roughness)+torch.rand(1,1,3,3)*roughness#*1#+1
    #img = img.to(device)
    #roughness
    #for n in range(8):
    #    roughness/=2.
    #    r = roughness# ** (n+1))
    #    print("roughness r=",r)
    #    img = grow_plasma_once(img,r)
    #print(img.size())
    #img = grow_plasma(img,.125)
    #img = grow_plasma(img,.0675)
    #img = grow_plasma(img, .03425)