from cvxopt.solvers import qp
from cvxopt.base import matrix
import matplotlib
#The following line is needed when using BASH in WINDOWS10
matplotlib.use('Agg') #no UI backend
import matplotlib.pylab as pylab
import numpy , random , math

# Linear Kernel Function
def linKernel(x1,x2):
	return numpy.dot(x1,x2) + 1

# Polynomial Kernel Function
def polyKernel(x1,x2,p):
	return math.pow(linKernel(x1,x2),p)

# Radial Kernel Function
def radKernl(x1,x2,sigma):
	return math.exp(-((numpy.dot(numpy.subtract(x1,x2), numpy.subtract(x1,x2))) / 2*math.pow(sigma,2)))

# Indicator function
def indFunction(x,y, param_list):
	sum = 0
	for i in range(len(param_list)):
		x_param     = param_list[i][0:2]
		t_param     = param_list[i][2]
		alpha_param = param_list[i][3]
		sum = sum + alpha_param*t_param*polyKernel((x,y),x_param,2)
	return sum



# Generate random data
# Uncomment the line below ("numpy.random") to generate
# the same dataset over and over again .
random.seed(100)
classA = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range (5)] + [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range (5)]
classB = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for i in range (10)]
#data is Nx3, where the 3 columns are for x, y and -1/+1
data = classA + classB
random.shuffle(data)

# Visualize data
pylab.figure(1)
pylab.hold(True)
pylab.plot([p[0] for p in classA], [p[1] for p in classA], 'bo')
pylab.plot([p[0] for p in classB], [p[1] for p in classB], 'ro')
pylab.title("Generated Datapoints")
# Instead of plotting, save the plot in a PNG file to review later
#pylab.show()
pylab.savefig("GeneratedData.png")

# Number of datapoints
N = len(data)

# WITHOUT SLACK VARIABLES#######################################################################################
# Define vectors and matrices for optimization
# P: NxN
# q: Nx1, full of "-1"
# h: Nx1, full of "0"
# G: NxN, with "-1" in main diagonal and "0" elsewhere

# P is given as P = t(i) * t(j) * K( x(i), x(j) ), where t is -1/+1 for binary classification
# t(i) or t(j) is the 3rd element of row i or j of data
P = numpy.zeros((N,N))
for i in range(N):
	for j in range(N):
		P[i][j] = data[i][2]*data[j][2]*polyKernel(data[i][0:2], data[j][0:2],2)

q = (-1)*numpy.ones(N)

h = ( 0)*numpy.ones(N)

G = (-1)*numpy.eye(N)


# Find alpha that optimizes the problem solution
r = qp(matrix(P), matrix(numpy.transpose(q)), matrix(G), matrix(h))
alpha = list(r['x'])

# Find non-zero alphas
# Define a floating-point threshold as zero
thres = pow(10,-5)
non_zero_alpha_list = []
for i in range(len(alpha)):
	if alpha[i]>thres:
		non_zero_alpha_list.append([data[i][0], data[i][1], data[i][2], alpha[i]])

# Plot boundaries
x_range = numpy.arange(-7, 7, 0.05)
y_range = numpy.arange(-7, 7, 0.05)

grid = matrix([[indFunction(x,y,non_zero_alpha_list) for y in y_range] for x in x_range])
pylab.contour(x_range, y_range, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths = (1, 3, 1))
pylab.title("Decision Boundaries Without Slack")

pylab.savefig("Results_Without_Slack.png")

# ADD SLACK VARIABLES###########################################################################################
# Define vectors and matrices for optimization
# h_s: 2Nx1, full of "0"
# G_s: 2NxN, with "-1" in main diagonal and "0" elsewhere
# C shows how much "slack" is allowed
C = pow(10,3)
h_s_upper = ( C)*numpy.ones(N)
h_s_lower = ( 0)*numpy.ones(N)
h_s = numpy.concatenate((h_s_upper, h_s_lower), axis=0)

G_s_upper = (+1)*numpy.eye(N,N)
G_s_lower = (-1)*numpy.eye(N,N)
G_s = numpy.concatenate((G_s_upper, G_s_lower), axis=0)

# Find alpha that optimizes the problem solution
r_s = qp(matrix(P), matrix(numpy.transpose(q)), matrix(G_s), matrix(h_s))
alpha_s = list(r_s['x'])

# Find non-zero alphas
# Define a floating-point threshold as zero and an upper limit as slack
low_thres  = pow(10,-5)
non_zero_alpha_list_slack = []
for i in range(len(alpha_s)):
	if (alpha_s[i]>=low_thres):# and (alpha_s[i]<=C):
		non_zero_alpha_list_slack.append([data[i][0], data[i][1], data[i][2], alpha_s[i]])

# Plot boundaries
x_range = numpy.arange(-7, 7, 0.05)
y_range = numpy.arange(-7, 7, 0.05)

grid_s = matrix([[indFunction(x,y,non_zero_alpha_list_slack) for y in y_range] for x in x_range])
# Visualize data
pylab.figure(2)
pylab.hold(True)
pylab.plot([p[0] for p in classA], [p[1] for p in classA], 'bo')
pylab.plot([p[0] for p in classB], [p[1] for p in classB], 'ro')
pylab.contour(x_range, y_range, grid_s, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths = (1, 3, 1))
pylab.title("Decision Boundaries With Slack")
pylab.savefig("Results_With_Slack.png")
