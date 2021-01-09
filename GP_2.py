import matplotlib.pyplot as plt
import numpy as np 
from scipy.optimize import minimize

def Rational_Quadratic_Kernel(X1,X2,alpha,length_scale):
    square_error=np.power(X1.reshape(-1,1)-X2.reshape(1,-1),2.0)
    kernel=np.power(1+square_error/(2*alpha*length_scale**2),-alpha)

    return kernel

def Calculate_init_Cov(Init_Cov,error,nums):
    Init_Cov+= 1/error*np.identity(nums)
    
    return Init_Cov

def Update(test,data_x,data_y,cov,error,alpha,length_scale):
    k=Rational_Quadratic_Kernel(data_x,test,alpha,length_scale)
    k_star=Rational_Quadratic_Kernel(test,test,alpha,length_scale)
    k_star = Calculate_init_Cov(k_star,error,500)
    #----update mean----#
    means=k.T @ np.linalg.inv(cov) @ data_y.reshape(-1,1)
    #----update var----#
    var=k_star-k.T @ np.linalg.inv(cov) @ k

    return means,var
    
def get_std_from_cov(Cov):
    std=[0.0]*500
    std=np.array(std)
    for i in range(len(test)):
        if Cov[i][i]<0:
            std[i]=0
        else:
            std[i]=float(Cov[i][i]**0.5)
            std[i]*=1.96
    return std

def objective_function(data_x,data_y,beta):
    def objective(theta):
        K = Rational_Quadratic_Kernel(data_x,data_x,alpha=theta[0],length_scale=theta[1])
        K = Calculate_init_Cov(K,error,34)
        L=np.linalg.cholesky(K)
        return 0.5*data_y.reshape(1,-1) @ np.linalg.inv(K) @ data_y.reshape(-1,1) + np.sum(np.log(np.diag(L))) + 0.5*len(data_x)*np.log(2*np.pi)

    return objective
"""
    data_x:存x , data_y:存y
    input.data=[x1,y1]
               [x2,y2]
                 ...
"""

#----Read intput.data----#
data_x, data_y = np.loadtxt('C:/Users/tim/Desktop/碩一/碩一下/ML/HW05/input.data', delimiter=' ', unpack=True)
#----Visualize Data----#
plt.figure()
plt.plot(data_x,data_y,'*',label='data point',color='black')




#----setting----#
test=np.linspace(-60,60,num=500)
error = 5
#----Optimize alpha & length_scale----#
objective_value=1e9
inits=[1e-2,1e-1,0,1e1,1e2]
#inits = [0.01,0.01,0.01,0.01,0.1,0]
#inits = [100,100,100,100,100]
#inits=[1e-2,1e-2,1e-2,1e-2,1e-2]
for init_alpha in inits:
    for init_length_scale in inits:
        res = minimize(objective_function(data_x,data_y,error),x0=[init_alpha,init_length_scale],bounds=((1e-5,1e5),(1e-5,1e5)))
        if res.fun<objective_value:
            objective_value = res.fun
            alpha_optimize,length_scale_optimize = res.x           

alpha = alpha_optimize
length_scale = length_scale_optimize
print("Optimize alpha:",alpha)
print("Optimize length scale",length_scale)

#----Adding error y=f(x)+ error~N(0,0.2)----#
Init_Cov = Rational_Quadratic_Kernel(data_x,data_x,alpha,length_scale)
Init_Cov = Calculate_init_Cov(Init_Cov,error,34)

#----Update mean & var----#
mean_predict,variance_predict = Update(test,data_x,data_y,Init_Cov,error,alpha,length_scale)
mean_predict = mean_predict.reshape(-1)
std_predict = get_std_from_cov(variance_predict)

#----Plot----#
plt.plot(test,mean_predict,label="prediction")
plt.fill_between(test,mean_predict+std_predict,mean_predict-std_predict,color='green',alpha=0.2)
plt.xlim(-60,60)
plt.show()