from numpy import *
from matplotlib import *
import matplotlib.pyplot as plt

# randomized data for task 3 and 4
train_x = linspace(1.0, 10.0, num=100)[:, newaxis]
train_y = sin(train_x) + 0.1*power(train_x, 2) + 0.5*random.randn(100, 1)

# randomized data for task 1, 2, 5, 6, 7
with load("TINY_MNIST.npz") as data:
	# images and answers for training data
	x, t = data["x"], data["t"]
	# images and answers for validation data
	x_eval, t_eval = data["x_eval"], data["t_eval"]


def part3():

	# Set data for parts 3.1 and 3.2 - pre-defined N and K values for learning
	N = [5, 50, 100, 200, 400, 800]
	K = [1, 3,  5,   7,   21,  101, 401]

	# Arrays to record number of errors for different N (task1) and K (task2) values
	task1_results = [0, 0, 0, 0, 0, 0]
	task2_results = [0, 0, 0, 0, 0, 0, 0]

	# For every image in the validation set, calculate the first nearest-neighbor with different training subsets (values in N)
	# Record only mismatches in task1_results
	for j in range(size(t_eval)):
		for i in range(size(N)):
			result = NearestNeighbor(1, x_eval[j], x[0:N[i]], t[0:N[i]])
			if result != int(t_eval[j]):
				task1_results[i] += 1
	print task1_results

	# For every image in the validation set, calculate the Kth-nearest-neighbor (values in K) with the full training set
	# Record only mismatches in task2_results
	for j in range(size(t_eval)):
		for i in range(size(K)):
			result = NearestNeighbor(K[i], x_eval[j], x, t)
			if result != int(t_eval[j]):
				task2_results[i] += 1

	print task2_results


def NearestNeighbor(k, v_image, training_set, training_ans):

	# The differences array, initialized to zero, will keep track of the total difference between pixels for each image
	differences = zeros((training_ans.shape))

	# The index is used to track the order of images examined
	index = 0

	# Iterate through every image in the training set, tracking the error in the differences array
	for image in training_set:
		local_dif = 0
		for i in range(size(image)):
			local_dif = local_dif + (image[i] - v_image[i])**2
		differences[index] = local_dif
		index += 1

	# Pair differences, calculated above, with the training answers - maps size of difference to a result
	vector_pairs = []
	for i in range(size(differences)):
		vector_pairs.append( (differences[i], training_ans[i]) )

	# Sort the vector_pairs array, so smallest differences appear first
	vector_pairs = sorted(vector_pairs)

	# Count the k-nearest neighbors, and add one to count[0] or count[1] accordingly
	count = [0, 0]
	for i in range(k):
		count[int(vector_pairs[i][1])] += 1

	# Return the value (0 or 1) which appears most in the k-nearest neighbors
	if count[0] > count[1]:
		return 0
	else:
		return 1


def linear_regression(data_matrix, output_vector, W, iterations, eta = 0.01, lda = 0):
	"""
		This function is used to calculate the linear regression given data, data_matrix, and 
		training the values for W. eta is the learning and lda is the lambda for weighing down predictions.
	"""
	dim = data_matrix.shape[1]
	T = data_matrix.shape[0]
	
	#Iterating over the iteration numbers
	for iteration in range(iterations):
		gradient = 0
		cost = 0
		for t in range(T):
			delta = output_vector[t] - f(data_matrix[t],W)
			cost += delta**2
			gradient = gradient + data_matrix[t] * delta
		#Setting the values of W, training
		W = W + (eta)*((2.0/T) * gradient - lda * W)
	return W


def f(x, omega):
	"""
		Function to make it easier read and calculate linear regression, given x and values for omega.
	"""
	return dot(x, omega)
	
def std_normalize(norm_array):
	"""
		Function to normalize a given array. By calculating the standard deviation
		and dividing the elements by it.
	"""
	global sigmas
	N = float(size(norm_array))
	mean = add.reduce(norm_array) / N
	sigma = 0.0
	for i in range(size(norm_array)):
		sigma = sigma + (norm_array[i] - mean)**2
	sigma = sigma / N
	sigma = sigma**0.5
	sigmas.append(sigma)
	
	#Avoiding divide by 0.
	if sigma != 0:
		norm_array = norm_array / sigma
	return norm_array

def calc_sigma(norm_matrix):
	"""
		Calculating the standard deviation for a given matrix.
	"""
	sigma = zeros((len(norm_matrix)))
	index = 0
	for row in norm_matrix:
		N = float(size(row))
		mean = add.reduce(row) / N
		sigma_1 = 0.0
		for i in range(size(row)):
			sigma_1 = sigma_1 + (row[i] - mean)**2
		sigma_1 = sigma_1 / N
		sigma_1 = sigma_1**0.5
		print sigma_1
		sigma[index] = sigma_1
		index += 1
	return sigma

def task3(k):
	"""
		Code for task 3. Using linear regression to predict underlying original data over k iterations.
		It outputs plots for the original data set and the linear prediction.
	"""
	T = size(train_x)
	omega = [0, 0]
	eta = 0.01
	
	#Iterating k times to train the solution.
	for q in range(k):
		my_sum = 0
		
		#Calculating omega[1]
		for i in range(size(train_x)):
			my_sum = my_sum + (train_y[i] - (omega[1]*train_x[i] + omega[0]))*train_x[i]
		omega[1] = omega[1] + (eta)*(2.0/T)*my_sum
		test_omega1.append(omega[1])
		print "OMEGA-1: " + str(omega[1])
		my_sum = 0
		
		#Calculating omega[0]
		for i in range(size(train_x)):
			my_sum = my_sum + (train_y[i] - (omega[1]*train_x[i] + omega[0]))
		omega[0] = omega[0] + (eta)*(2.0/T)*my_sum
		test_omega0.append(omega[0])
	
	#Plotting the required plots.
	l1 = plt.plot(train_x,train_y, 'ro', label='Original Training Dataset')
	yout = train_x*omega[1] + omega[0]
	p1 = plt.plot(train_x, yout, label = 'Linear Fit')
	plt.legend(loc = 'upper left')
	plt.ylabel('Y (y of Artifical DS)')
	plt.xlabel('X (x of Artifical DS)')
	plt.title('Task3 Linear Fit Plot', fontsize = 14)
	plt.show()


sigmas = []
omega = []
def task4(training_data, dim, iterations):
	# Create a normalized array of dim dimensions
	global sigmas
	global omega
	sigmas = []
	omega = []
	data_matrix = []
	
	for d in range(dim+1):
		data_matrix.append(training_data**d)
	norm_matrix = [std_normalize(row) for row in data_matrix]
	norm_matrix = array(norm_matrix)
	
	# Initialized variables for gradient descent
	W = zeros(dim)
	omega = linear_regression(norm_matrix, train_y, W, iterations)
	sigmas = array(sigmas)
	sigmas = sigmas.ravel()
	sigmas[sigmas == 0] = 1
	omega = omega/sigmas
	plot_y = omega[0]*(training_data**0)
	
	for d in range(1, dim+1):
		plot_y = plot_y + omega[d]*(training_data**d)
	
	#Plotting training data and plot.
	plt.plot(training_data, train_y)
	plt.plot(training_data, plot_y)
	plt.ylabel('Y (y of Artifical DS)')
	plt.xlabel('X (x of Artifical DS)')
	plt.title('Task3 Linear Fit Plot', fontsize = 14)
	plt.show()


def task5(iterations = 10000, dim = 64, batchSize = 50):
	"""
		This function is for task 5. Using stochastic linear regression to classify images as
		3 or 5.
	"""
	# Set pre-defined N value for minibatch learning
	N = [50]
	x65 = array([append(1,q) for q in x])

	for n in N:
		W = zeros(dim+1)
		#count = 0.0
		for iteration in range(iterations):
			vecount = 0.0
			tecount = 0.0
			for batchIndex in range(n/batchSize):
				
				batch = x65[batchIndex * batchSize: batchIndex * batchSize + batchSize]		
				output_vector = t[batchIndex * batchSize : batchIndex * batchSize + batchSize]			
				W = linear_regression(batch,output_vector,W,1)

			
			#Getting training errors
			for i,image in enumerate(x65[:n]):
				
				predict = 0
				if f(image,W) > 0.5:
					predict = 1
				if predict != t[i]:
					tecount += 1			

			#Calculating validation errors
			for i,image in enumerate(x_eval):
				predict = 0
				if f(append(1,image),W) > 0.5:
					predict = 1
				if predict != t_eval[i]:
					vecount += 1
					
			#Printing the number of epochs : the iteration number : Number of Training Errors : Number of Validation Errors
			print n, ' : ',iteration, ' : ', (tecount/n)*400, ' : ',(vecount/len(t_eval))*400

def task6(dim = 64, epochs = 20000, batchSize = 50):
	# Set pre-defined N value (50) for minibatch learning, but with multiple epochs
	N = 50
	W = zeros(dim+1)
	x65 = array([append(1,q) for q in x])
	
	#Placeholder lists to hold plotting values.
	err = []
	terr = [] 
	verr = []
	
	for e in range(epochs):
		#Variables to hold error counts
		vecount = 0.0
		tecount = 0.0

		
		for batchIndex in range(N/batchSize):
			#creating minibatches
			batch = x65[batchIndex * batchSize: batchIndex * batchSize + batchSize]		
			output_vector = t[batchIndex * batchSize : batchIndex * batchSize + batchSize]
			oldW = W
			W = linear_regression(batch,output_vector,W,1, eta = 0.01)

		#Get training errors
		for i,image in enumerate(x65[:50]):
			predict = 0
			if f(image,W) > 0.5:
				predict = 1
			if predict != t[i]:
				tecount += 1
			
		#Getting Validation Errors
		for i,vimage in enumerate(x_eval):
			predict = 0
			if (f(vimage,W[1:])+W[0]) > 0.5:
				predict = 1
			if predict != t_eval[i]:
				vecount += 1
		
		#Storing value of errors every 100 epochs
		if (e+1)%100 == 0:
			err.append(e+1)
			terr.append(tecount/len(t))
			verr.append(vecount/len(t_eval))
	
	#Creating numpy arrays and plotting
	err = array(err)
	terr = array(terr)
	verr = array(verr)
	plt.plot(err,(terr*400), label = "Number of Epochs vs Training Errors")
	plt.plot(err,(verr*400), label = "Number of Epochs vs Validation Errors")
	plt.title("Task 6: Epochs vs Errors")
	plt.legend(loc = 'upper left')
	plt.show()

def task7(dim = 64, epochs = 50000, batchSize = 50):
	# Set pre-defined N value (50) for minibatch learning, but with multiple epochs
	N = [50]#, 100, 200, 400, 800]
	L = [0, 0.0001, 0.001, 0.01, 0.1, 0.5]
	W = zeros(dim+1)
	x65 = array([append(1,q) for q in x])
	vimages = array([append(1,v) for v in x_eval])

	for l in L:		
		for e in range(epochs):		
			vecount = 0.0
			tecount = 0.0

			for batchIndex in range(N[0]/batchSize):
				batch = x65[batchIndex * batchSize: batchIndex * batchSize + batchSize]		
				output_vector = t[batchIndex * batchSize : batchIndex * batchSize + batchSize]
				oldW = W
				W = linear_regression(batch,output_vector,W,1, eta = 0.01, lda = l)

			#Calculating training errors
			for i,image in enumerate(x65[:50]):
				predict = 0
				if f(image,W) > 0.5:
					predict = 1
				if predict != t[i]:
					tecount += 1
			
			#Calculating validation errors
			for i,vimage in enumerate(vimages):
				predict = 0
				if f(vimage,W) > 0.5:
					predict = 1
				if predict != t_eval[i]:
					vecount += 1

			if (e+1)%100 == 0:
				#Printing the Lambda Value : Iteration in Epoch : Number of Training Errors : Number of Validation Errors
				print l, ' : ',e+1, ' : ', (tecount/len(t))*800,' : ',(vecount/len(t_eval))*400