import numpy.linalg as lin
def descentLineSearch(F, descent, ls, alpha0, x0, tol, maxIter):
	"""
	Steepest Descent Line Search algorithm modified from tutorial 2

	Inputs
		 F: dictionary with fields
		   - f: function handler
		   - df: gradient handler
		   - d2f: Hessian handler
		 descent: specifies descent direction {'steepest', 'newton'}
		 ls: function handle for computing the step length
		 alpha0: initial step length
		 x0: initial iterate
		 tol: stopping condition on minimal allowed step
		      norm(x_k - x_k_1)/norm(x_k) < tol;
		 maxIter: maximum number of iterations
	OUTPUTS
		 xMin, fMin: minimum and value of f at the minimum
		 nIter: number of iterations
		 info: dictionary with information about the iteration
		   - "xs": iterate history
		   - "alphas": step lengths history

	"""
	# Initialization
	nIter = 0
	x_k = x0
	info = {}
	info["xs"] = [x0]
	info["alphas"] = [alpha0]
	stopCond = False

	# Loop until convergence or maximum number of iterations
	while (~stopCond and nIter <= maxIter):
		# Increment iterations
		nIter += 1

		# p_k steepest descent	direction
		p_k = -F ["df"] (x_k)

		# Call line search given by handle ls to compute step length alpha_k
		alpha_k = ls (x_k, p_k, alpha0)

		# Update x_k
		x_k_1 = x_k
		x_k = x_k + alpha_k * p_k

		# Store iteration info
		info["xs"].append(x_k)
		info["alphas"].append(alpha_k)

		stopCond = (lin.norm (F["df"] (x_k), 'inf') < tol * (1 + abs (F["f"] (x_k))))

	xMin = x_k
	fMin = F["f"] (x_k)

	return xMin, fMin, nIter, info

