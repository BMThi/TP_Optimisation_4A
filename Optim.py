import numpy as np
import scipy.linalg

def deriv_num(J,a,d,compute_grad=True,compute_Hess=True) :
    """test numerically the derivative and the Hessian of a function.
        
    Parameters
    ----------
    J : instance of a class
        The function to be tested it must have the following methods, where x is a 1d vector
        of size n
            -- J.eval(x) : evaluation of J at point x, must return a float
            -- J.grad(x) : evaluation of the gradient of J at point x, must a 1d vector of size n
            -- J.Hess(x) : evaluation of the Hessian of J at point x, typically a n*n matrix
    a : 1d vector of size n
        Point at which the numerical derivatives are evaluated
    d : 1d vector of size n
        Direction in which the numerical derivatives are evaluated
    compute_grad : Boolean
        Flag that tests the function J.grad against numerical derivatives
    compute_Hess : Boolean
        Flag that tests the function J.Hess against numerical derivatives of J.grad
    
   Output 
   -----
   This function does not have an output, it prints a string s.
    """
        
    eps_range=[0.1**(i+1) for i in range(12)]
    for eps in  eps_range:
        s='eps {:1.3e}'.format(eps)
        if compute_grad :
            # ratio of numerical derivatives of J and prediction given by J.grad
            ratio=(J.value(a+eps*d)-J.value(a))/(eps*np.dot(J.grad(a),d)) 
            s+=' grad {:1.1e}'.format(np.abs(ratio-1)) 
        if compute_Hess :
            v1=(J.grad(a+eps*d)-J.grad(a))/eps # numerical derivative of J.grad
            v2=J.Hess(a).dot(d)  # prediction given by J.Hess
            ratio=np.linalg.norm(v1)/np.linalg.norm(v2) #norm ratio
            angle=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)) # cosinus of the angle between the vectors
            s+=' ratio {:1.1e}'.format(np.abs(ratio-1.))
            s+=' angle {:1.1e}'.format(angle-1.)
        print(s)
        
def main_algorithm(function,step,xini,dc,ls,itermax = 20000,tol=1.e-4,verbose=True):
    """Perform a minimization algorithm of a function
    
    Parameters
    ----------
    function : instance of a class
        The function to be minimized, depending on the choice of linesearch and direction of descent,
        it must have the following methods, where x is a 1d vector of size n
            -- function.eval(x) : evaluation of J at point x, must return a float
            -- function.grad(x) : evaluation of the gradient of J at point x, must a 1d vector of size n
            -- function.Hess(x) : evaluation of the Hessian of J at point x, typically a n*n matrix
    step : positive float
        Initial guess of the step
    xini : 1d vector of size n
        initial starting point
    dc : callable
        descent,info_dc=dc(x,function,df,res)
        computes the descent direction with parameters 
           -x: the point x
           -df : the gradient of function at point x
           -function : the function
        The function dc returns
            -descent : the direction of descent
            -info_dc : information about the behavior of the function dc   
    ls : callable
        x2,f2,df2,step2,info_ls=ls(x, function, step, descent,f,df)
        performs a line search, the parameters are
           -x : initial point
           -step : initial step
           -function : the function to be minimized
           -f,df : the values of function(x) and the gradient of the function of x
           -descent : the descent direction
        the function returns
            -x2 : the new point x+step2*descent
            -f2 : the value of function at point x2
            -df2 : the value of the gradient of the function at point x2
            -step2 : the step given by the linesearch
            -info_ls : some information about the behavior of the function ls
    itermax : int
        maximum number of iterations
    tol : float
       stopping criterion
    verbose : Boolean
        Printing option of the algorithm
    
    Returns
    --------
    The function returns a single dictionary res, the entries are
    res['list_x'] : list of 1d vectors of size n which are the iterates points of the algorithm
    res['list_steps'] : list of positive floats which are the different steps
    res['list_grads'] : list of positive floats which are the value of the euclidean norm of the gradients of the function
    res['final_x'] : 1d vector, final value of x
    res['dc'] : list of the different infos given by the functions dc
    res['ls'] : list of the different infos given by the functions ls        
    """
    x = xini
    res={'list_x':[],'list_steps':[],'list_costs':[],'list_grads':[],'final_x':[],'dc':[],'ls':[]}
    nbiter = 0
    f=function.value(x)
    df= function.grad(x)
    err=np.linalg.norm(df)
    if verbose :  print('iter={:4d} f={:1.3e} df={:1.3e} comp=[{:4d},{:4d},{:4d}]'.format(nbiter,f,err,function.nb_eval,function.nb_grad,function.nb_hess))
    res['list_x'].append(x.copy())
    res['list_costs'].append(f)
    res['list_grads'].append(err)
    while (err > tol) and (nbiter < itermax):
        descent,info_dc = dc(x, function,df)
        x,f,df,step,info_ls = ls(x, function, step, descent,f,df)
        err = np.linalg.norm(df)
        res['list_x'].append(x.copy())
        res['list_costs'].append(f)
        res['list_grads'].append(err)
        res['list_steps'].append(step)
        res['dc'].append(info_dc)
        res['ls'].append(info_ls)
        nbiter+=1
        if verbose : print('iter={:4d} f={:1.3e} df={:1.3e} comp=[{:4d},{:4d},{:4d}]'.format(nbiter,f,err,function.nb_eval,function.nb_grad,function.nb_hess))
        if (err <= tol):
            res['final_x']=np.copy(x)
            if verbose : print("Success !!! Algorithm converged !!!")
            return res
    if verbose : print("FAILED to converge")
    return res       

def dc_gradient(x,function,df) :
    """Choice of direction of descent : GRADIENT METHOD
    
    Parameters
    ----------
        x : 1d vector of size n
            actual iterate of the method
        function : instance of a class
            The function to be minimized
        df : The actual value of the gradient
          
    returns
    -------
       descent : 1d vector of size n
           direction of descent 
       ls_info : 
           Information about the behavior of the function
    """
    descent=-df
    ls_info=None
    return descent,ls_info

def dc_newton(x, function, df):
    """Choice of direction of descent : NEWTON ALGORITHM
    
    Parameters
    ----------
        x : 1d vector of size n
            actual iterate of the method
        function : instance of a class
            The function to be minimized
        df : The actual value of the gradient
          
    returns
    -------
       d : 1d vector of size n
           Newton's direction of descent 
       ls_info : 
           Information about the behavior of the descent process
    """

    H_x_inv = np.linalg.inv(function.Hess(x))

    # Newton's direction
    d = np.dot(-H_x_inv, df)

    # Compute the cosine of the angle between d and -grad
    cos_theta = np.dot(d, -df) / (np.linalg.norm(d) * np.linalg.norm(df))

    # Use Newton step if cos(theta) > 0.1, otherwise use negative gradient
    d, ls_info = (d, "Newton") if cos_theta > 0.1 else (-df, "gradient")

    return d,ls_info

def ls_constant(x, function, step, descent,f,df) :
    """Line search : FIXED STEP
    
    Parameters
    ----------
        x : 1d vector of size n
            actual iterate of the method
        function : instance of a class
            The function to be minimized
        step : float
            The starting guess of the step
        descent : 1d vector of size n
            The descent direction
        f : float
            the value of the function at point x
        df : 1d vector of size n
            the gradient of the function at point x
          
    returns
    -------
        x2 : 1d vector of size n
            x2=x+step2*descent
        f2 : float
            the value of the function at point x2
        df2 : 1d vector of size n
            the gradient of the function at point x2
        step2 : float
            The step chosen by the method
        info : Information about the behavior of the method 
       
    """
    step2=step
    x2=x+step2*descent
    f2=function.value(x2)
    df2=function.grad(x2)
    info=None
    return x2,f2,df2,step2,info

def ls_backtracking(x, function, step, descent,f,df) :
    """Line search : BACKTRACKING
    
    Parameters
    ----------print(res['dc'])
        x : 1d vector of size n
            actual iterate of the method
        function : instance of a class
            The function to be minimized
        step : float
            The starting guess of the step
        descent : 1d vector of size n
            The descent direction
        f : float
            the value of the function at point x
        df : 1d vector of size n
            the gradient of the function at point x
          
    returns
    -------
        x2 : 1d vector of size n
            x2=x+step2*descent
        f2 : float
            the value of the function at point x2
        df2 : 1d vector of size n
            the gradient of the function at point x2
        step2 : float
            The step chosen by the method
        info : Information about the behavior of the method 
       
    """
    step2=step
    x2=x+step2*descent
    f2=function.value(x2)
    while f2>f : 
        step2*=0.5
        x2=x+step2*descent
        f2=function.value(x2)
    df2=function.grad(x2)
    info=None
    return x2,f2,df2,step2,info

def ls_partial_linesearch(x, function, step, descent,f,df) :
    """Line search : PARTIAL LINESEARCH
    
    Parameters
    ----------
        x : 1d vector of size n
            actual iterate of the method
        function : instance of a class
            The function to be minimized
        step : float
            The starting guess of the step
        descent : 1d vector of size n
            The descent direction
        f : float
            the value of the function at point x
        df : 1d vector of size n
            the gradient of the function at point x
          
    returns
    -------
        x2 : 1d vector of size n
            x2=x+step2*descent
        f2 : float
            the value of the function at point x2
        df2 : 1d vector of size n
            the gradient of the function at point x2
        step2 : float
            The step chosen by the method
        info : Information about the behavior of the method 
       
    """
    S = np.array([0.1, 0.5, 1, 2, 10]) * step  # Candidate step sizes
    # Evaluate the function at different points along the line and choose the step size with the minimum value.
    f2_array = np.array([function.value(x + s * descent) for s in S])
    step2 = S[np.argmin(f2_array)]
    x2 = x + step2 * descent
    f2 = np.min(f2_array)
    df2 = function.grad(x2)
    info = None  
    return x2, f2, df2, step2, info

def ls_wolfe(x, function, step, descent, f, df, epsilon1=1.e-4, epsilon2=0.9):
    """Line search using Wolfe conditions.
    
    Parameters
    ----------
        x : 1d vector of size n
            Current iterate of the method.
        function : instance of a class
            The function to be minimized.
        step : float
            Initial guess of the step.
        descent : 1d vector of size n
            The descent direction.
        epsilon1 : float, optional
            Parameter for the Wolfe condition 1 (default is 1e-4).
        epsilon2 : float, optional
            Parameter for the Wolfe condition 2 (default is 0.9).
        max_iter : int, optional
            Maximum number of iterations (default is 100).
    
    returns
    -------
        x2 : 1d vector of size n
            x2=x+step2*descent
        f2 : float
            the value of the function at point x2
        df2 : 1d vector of size n
            the gradient of the function at point x2
        step2 : float
            The step chosen by the method
        info : Information about the behavior of the method 
       
    """
    info = 0
    step2 = step
    s_minus = 0
    s_plus = np.inf
    x2=x+step2*descent
    f2=function.value(x2)
    df2=function.grad(x2)

    while f2 > f + epsilon1 * step2 * np.dot(df, descent) or \
        np.dot(df2, descent) < epsilon2 * np.dot(df, descent):

        # Wolfe Condition 1: Avoid big step
        if f2 > f + epsilon1 * step2 * np.dot(df, descent):
            s_plus = step2
            step2 = 0.5 * (s_minus + s_plus)
            
        # Wolfe Condition 2: Avoid small step
        else:
            s_minus = step2
            if s_plus < np.inf:
                step2 = 0.5 * (s_minus + s_plus)
            else:
                step2 *= 2

        x2 = x + step2 * descent
        f2 = function.value(x2)
        df2 = function.grad(x2)
        info += 1  
    return x2, f2, df2, step2, info

def ls_wolfe_step_is_one(x,function,step,descent,f,df) :
    return ls_wolfe(x,function,1.,descent,f,df)

class BFGS:
    def __init__(self, nb_stock_max=8):
        """
        Initialize the BFGS optimizer.

        Parameters:
        - nb_stock_max: int, optional
            Maximum number of iterations to be stored in the history (default is 8).
        """
        self.nb_stock_max = nb_stock_max
        self.stock = []
        self.last_iter = []

    def push(self, x, grad):
        """
        Update the history with the current iterate and gradient, and store (s, g, rho) in the history.

        Parameters:
        - x: 1d vector
            Current iterate.
        - grad: 1d vector
            Gradient at the current iterate.
        """
        if len(self.last_iter) > 0:
            x_prev, grad_prev = self.last_iter
            s = x - x_prev
            y = grad - grad_prev
            rho = 1 / np.dot(s, y)

            if rho > 0:
                if len(self.stock) > self.nb_stock_max:
                    self.stock.pop(0)
                self.stock.append((s, y, rho))
                self.last_iter = (np.copy(x),np.copy(grad))
            else:
                # If rho is non-positive, something went wrong, clear the stock
                self.stock = []
        else:
            # If last_iter is empty, simply update it
            self.last_iter = (np.copy(x),np.copy(grad))

    def get(self, grad):
        """
        Modify the descent direction and apply the BFGS algorithm.

        Parameters:
        - grad: 1d vector
            Gradient at the current iterate.
            
        Returns:
        - r: 1d vector
            Resulting descent direction.
        """
        r = -np.copy(grad)
        l_alpha = []
        if len(self.stock) > 0:
            for s, y, rho in reversed(self.stock):
                alpha = rho * np.dot(s, r)
                r = r - alpha * y
                l_alpha.append(alpha)

            s_kmin, y_kmin, rho_kmin = self.stock[0]
            r = (np.dot(s_kmin, y_kmin) / np.dot(y_kmin, y_kmin)) * r

            for ((s, y, rho), alpha) in zip(self.stock, reversed(l_alpha)):
                beta = rho * np.dot(y, r)
                r = r + (alpha - beta) * s

        return r

    def dc(self, x, function, df):
        """
        Apply the BFGS algorithm to modify the descent direction.

        Parameters:
        - x: 1d vector
            Current iterate.
        - function: instance of a class
            The function to be minimized.
        - df: 1d vector
            Gradient at the current iterate.

        Returns:
        - r: 1d vector
            Resulting descent direction.
        - info: str
            Information about the behavior of the BFGS algorithm.
        """
        self.push(x, df)
        r = self.get(df)
        info = "L-BFGS Direction Computed."
        return r, info
