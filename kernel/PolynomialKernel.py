from kernel.Kernel import Kernel


class PolynomialKernel(Kernel):
    def __init__(self, degree=2, theta=1.0):
        Kernel.__init__(self)
        self.degree = degree
        self.theta = theta
        
    def __str__(self):
        s=self.__class__.__name__+ "=["
        s += "degree="+ str(self.degree)
        s += ", " + Kernel.__str__(self)
        s += "]"
        return s
    
    def kernel(self, X, Y=None):
        """
        Computes the polynomial kerpy k(x,y)=(1+theta*<x,y>)^degree for the given data
        X - samples on right hand side
        Y - samples on left hand side, can be None in which case its replaced by X
        """
        if Y is None:
            Y = X
        
        return pow(self.theta+X.dot(Y.T), self.degree)
