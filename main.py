import numpy as np
from sympy import symbols, diff
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d



iterations = 0      
vector=[40,50] #original "vector"
learning_rate = .1


x,y = symbols('x y', real = True)

z = x**2+y**2  # OUR FUNCTION


vector_a = diff(z, x) # calculates the partial in respect to x
vector_b = diff(z, y) # calculates the partial in respect to y


#are you looking at this, ive struck gold
def calc_gradient_x(x_a,y_a): #evaluate the x partial component of the gradient at x
    evaluated = vector_a.subs(x, x_a)
    evaluated_2 = evaluated.subs(y, y_a)
    return evaluated_2


def calc_gradient_y(x_a,y_a): #evaluates the y partial component of the gradient at x
  evaluated = vector_b.subs(x, x_a)
  evaluated_2 = evaluated.subs(y, y_a)
  return evaluated_2

def f(x_1, y_1):
    evaluated = z.subs(x, x_1)
    evaluated_2 = evaluated.subs(y, y_1)
    return evaluated_2
# gradient_descentio():


z_value = f(vector[0], vector[1])
  
historia_x = [(vector[0]),]
historia_y = [(vector[1]),]
historia_z = [z_value]
  
l = 3 #inputs
d = 51 # sets


z_value = f(vector[0], vector[1])


while iterations <50:#is the iterations 
  
    print ("Iteration", iterations)
    print("Starting Point:", vector)
    kangaroo = [vector[0], vector[1]]
    direction_x = calc_gradient_x(vector[0], vector[1]) #calculates x component of the gradient

    direction_y = calc_gradient_y(vector[0], vector[1]) #calculates y component of the gradient

    direction = [direction_x, direction_y] #stores the x and y component of the gradient in an array so we could then multiple this by the learning rate

    stepsize = [x_y * learning_rate for x_y in direction] #  caclulates the stepsize(gradient * learning rate)
  #vector = vector - learning_rate*gradient(vector) #I call on the gradient function
    vector = np.subtract(vector, stepsize) #subtracts current position with the step size this equals the new position after taking a step

    z_value = f(vector[0],vector[1])
    print(z_value)
#d2_array = [3][50]
    
    print("End Point: ", vector)
    print("")
    iterations = iterations + 1 #keeps count of the iteration counter

    historia_x.append(vector[0])
    historia_y.append(vector[1])
    historia_z.append(z_value)
  
    print()
    kangaroo_newpos = np.sum(np.subtract(kangaroo, vector))
    #print("Kangaroo", kangaroo_newpos)
    if kangaroo_newpos < 0.01:
      break
        #breaks optimization when you meet a level of precision
#smartyy
  

print("Final optimized inputs,", vector)

historia_x = np.array(historia_x)
historia_y = np.array(historia_y)
historia_z = np.array(historia_z)

#actually it works fine oops

#     THIS IS FOR THE GRAPH      ----------------------------------------

xBound = 2
yBound = 2
Y = np.arange(-xBound, xBound, .078)
X = np.arange(-yBound, yBound, .078) 
X, Y = np.meshgrid(X, Y)
Z= X**2 + Y**2

fig = plt.figure(figsize=(12, 7)) #figsize=(x, y) (in inches)
ax1 = fig.add_subplot(121, projection = '3d')#creates two plots

#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

surf = ax1.plot_surface(X,Y,Z, rstride=1, cstride=1, alpha=.8, linewidth=0, cmap=cm.jet,antialiased=True)
ax1.set_xlabel("X axis")
ax1.set_ylabel("Y axis")
ax1.set_zlabel("Z axis")


#color options: coolwarm, hot, jet, cool, hsv, spectral, plasma, spring, tab20, viridis

#plt.show()

#-------plots path of optimization--------

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
ax2 = fig.add_subplot(122, projection = '3d')
#unicorn_barf = LinearSegmentedColormap= ("green", "red")

# ax.contour3D is used plot a contour graph


ax2.scatter(historia_x, historia_y,historia_z, c='purple')
ax2.set_xlabel("X axis")
ax2.set_ylabel("Y axis")
ax2.set_zlabel("Z axis")
#plt.scatter(testing_x, testing1, Z, c=z , alpha= 1)


plt.tight_layout()#adjusts space between subplots to avoid overlap

plt.show()
