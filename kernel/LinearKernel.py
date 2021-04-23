from kernel.Kernel import Kernel


class LinearKernel(Kernel):
    def __init__(self, is_sparse=False):
        Kernel.__init__(self)
        self.is_sparse = is_sparse

    def __str__(self):
        s=self.__class__.__name__+ "=["
        s += "" + Kernel.__str__(self)
        s += "]"
        return s
    
    def kernel(self, X, Y=None):
        """
        Computes the linear kerpy k(x,y)=x^T y for the given data
        X - samples on right hand side
        Y - samples on left hand side, can be None in which case its replaced by X
        """

        if Y is None:
            Y = X
        if self.is_sparse:
            return X.dot(Y.T).todense()
        else:
            return X.dot(Y.T)
