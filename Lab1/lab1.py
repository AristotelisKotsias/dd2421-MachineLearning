import monkdata as m
import dtree as dt
import drawtree_qt4 as drawt
import random
import numpy as np
import pyqtgraph as pg

#Partitioning function
def partition(data, fraction):
	ldata = list(data)
	random.shuffle(ldata)
	breakPoint = int(len(ldata) * fraction)
	return ldata[:breakPoint], ldata[breakPoint:]

#---------------------------------------------------------
#ASSIGNMENTS 1-2##########################################
#---------------------------------------------------------
print "---------------"
print "ASSIGNMENTS 1-2"
print "---------------"
print "Entropies:"
entropy1 = dt.entropy(m.monk1)
entropy2 = dt.entropy(m.monk2)
entropy3 = dt.entropy(m.monk3)
print entropy1
print entropy2
print entropy3
print "\n"
wait = raw_input("PRESS ENTER TO CONTINUE.")
#---------------------------------------------------------
#ASSIGNMENTS 3-4##########################################
#---------------------------------------------------------
print "---------------"
print "ASSIGNMENTS 3-4"
print "---------------"
print "Monk1 Information Gain:\n"
for i in range (0,6):
	print dt.averageGain(m.monk1, m.attributes[i])
print "\n"
print "Monk2 Information Gain:\n"
for i in range (0,6):
	print dt.averageGain(m.monk2, m.attributes[i])
print "\n"
print "Monk3 Information Gain:\n"
for i in range (0,6):
	print dt.averageGain(m.monk3, m.attributes[i])

#As an example, we check Information Gain of Monk1, thus
#we select a5 (or m.attributes[4]) because of highest gain
sub1 = dt.select(m.monk1, m.attributes[4], 1)
sub2 = dt.select(m.monk1, m.attributes[4], 2)
sub3 = dt.select(m.monk1, m.attributes[4], 3)
sub4 = dt.select(m.monk1, m.attributes[4], 4)

print "Information Gain of all subsets of MONK1:"
for i in range (0,6):
	print dt.averageGain(sub1, m.attributes[i])
for i in range (0,6):
	print dt.averageGain(sub2, m.attributes[i])
for i in range (0,6):
	print dt.averageGain(sub3, m.attributes[i])
for i in range (0,6):
	print dt.averageGain(sub4, m.attributes[i])

print "\n"
wait = raw_input("PRESS ENTER TO CONTINUE.")
#print dt.mostCommon(sub2)


#---------------------------------------------------------
#ASSIGNMENT 5#############################################
#---------------------------------------------------------
print "---------------"
print "ASSIGNMENT 5"
print "---------------"

t1   = dt.buildTree(m.monk1, m.attributes)
t2   = dt.buildTree(m.monk2, m.attributes)
t3   = dt.buildTree(m.monk3, m.attributes)

#Find error
print "\nTrain and Test error for MONK1 tree:"
err1  = 1-dt.check(t1,m.monk1) 
print(err1)
err1t = 1-dt.check(t1,m.monk1test)
print(err1t)
#drawt.drawTree(t1)

print "\nTrain and Test error for MONK2 tree:"
err2  = 1-dt.check(t2,m.monk2) 
print(err2)
err2t = 1-dt.check(t2,m.monk2test)
print(err2t)

print "\nTrain and Test error for MONK3 tree:"
err3  = 1-dt.check(t3,m.monk3) 
print(err3)
err3t = 1-dt.check(t3,m.monk3test)
print(err3t)

print("")
wait = raw_input("PRESS ENTER TO CONTINUE.")


#---------------------------------------------------------
#ASSIGNMENTS 6-7##########################################
#---------------------------------------------------------
print "---------------"
print "ASSIGNMENTS 6-7"
print "---------------"


#Specify how many iterations are needed for statistics
iterations = 100;

#Specify (monktrain/monk) proportion (0.3 to 0.8)
split      = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

#Initialize the test errors for each MONK dataset
#"[0]*iterations" creates an 1D-array of zeros with length "iterations"
test_err1  = [1]*iterations
test_err2  = [1]*iterations
test_err3  = [1]*iterations

#Initialize mean values
mean1      = [0]*len(split)
mean2      = [0]*len(split)
mean3      = [0]*len(split)

#Initialize variances
var1       = [0]*len(split)
var2       = [0]*len(split)
var3       = [0]*len(split)

#Initialize standard deviations
std1       = [0]*len(split)
std2       = [0]*len(split)
std3       = [0]*len(split)

#This loop is to cover all the "split" values
for i in range (0,len(split)):
	print "\nRunning for %.1f split, please wait...\n" %split[i]
	#This loop is for the number of desired of iterations
	for j in range (0, iterations):
		#Separate data into test and validation chunks
		monk1train, monk1val = partition(m.monk1, split[i])
		monk2train, monk2val = partition(m.monk2, split[i])
		monk3train, monk3val = partition(m.monk3, split[i])

		#Build trees using ONLY train data
		t1t   = dt.buildTree(monk1train, m.attributes)
		t2t   = dt.buildTree(monk2train, m.attributes)
		t3t   = dt.buildTree(monk3train, m.attributes)

		#MONKDATA1################################################
		best_tree_updated = True

		#Start pruning
		while(best_tree_updated):

			best_tree_updated = False
			
			#Prune tree
			t1t_pruned = dt.allPruned(t1t)

			#Find number of alternative trees
			len_t1t_pruned = len(t1t_pruned)
			
			#Find the best tree that has the lowest error of all
			for k in range (0,len_t1t_pruned):
				curr_err = 1-dt.check(t1t_pruned[k], monk1val)
				if (curr_err < test_err1[j]):
					#best_tree = k
					test_err1[j] = curr_err
					best_tree_updated = True

		mean1[i] = mean1[i] + test_err1[j]

		#MONKDATA2################################################
		best_tree_updated = True

		#Start pruning
		while(best_tree_updated):

			best_tree_updated = False
			
			#Prune tree
			t2t_pruned = dt.allPruned(t2t)

			#Find number of alternative trees
			len_t2t_pruned = len(t2t_pruned)
			
			#Find the best tree that has the lowest error of all
			for k in range (0,len_t2t_pruned):
				curr_err = 1-dt.check(t2t_pruned[k], monk2val)
				if (curr_err < test_err2[j]):
					#best_tree = k
					test_err2[j] = curr_err
					best_tree_updated = True

		mean2[i] = mean2[i] + test_err2[j]

		#MONKDATA3################################################
		best_tree_updated = True

		#Start pruning
		while(best_tree_updated):

			best_tree_updated = False
			
			#Prune tree
			t3t_pruned = dt.allPruned(t3t)

			#Find number of alternative trees
			len_t3t_pruned = len(t3t_pruned)

			#Find the best tree that has the lowest error of all
			for k in range (0,len_t3t_pruned):
				curr_err = 1-dt.check(t3t_pruned[k], monk3val)
				if (curr_err < test_err3[j]):
					#best_tree = k
					test_err3[j] = curr_err
					best_tree_updated = True

		mean3[i] = mean3[i] + test_err3[j]

	#Find mean value of error for each split value

	mean1[i] = mean1[i]/iterations
	mean2[i] = mean2[i]/iterations
	mean3[i] = mean3[i]/iterations
	print "Mean Test-Error\n"
	print mean1[i], mean2[i], mean3[i]


	#Find variance of error for each split value
	mean_square_error1 = 0.0
	mean_square_error2 = 0.0
	mean_square_error3 = 0.0
	
	for j in range (0, iterations):
		mean_square_error1 = mean_square_error1 + pow(test_err1[j],2.0)
		mean_square_error2 = mean_square_error2 + pow(test_err2[j],2.0)
		mean_square_error3 = mean_square_error3 + pow(test_err3[j],2.0)
	
	var1[i] = mean_square_error1/iterations - pow(mean1[i],2.0)
	var2[i] = mean_square_error2/iterations - pow(mean2[i],2.0)
	var3[i] = mean_square_error3/iterations - pow(mean3[i],2.0)
	
	print "\nTest-Error Variance"
	print var1[i],var2[i],var3[i]

	#Find standard deviation
	std1[i] = pow(var1[i],0.5)
	std2[i] = pow(var2[i],0.5)
	std3[i] = pow(var3[i],0.5)

	print "\nTest-Error Standard Deviation\n"
	print std1[i],std2[i],std3[i]

#Change background to white
pg.setConfigOption('background', 'w')           

#Create graph of means
graph1 = pg.plot(split,mean1,symbol='o',title="MONKDATA1, Mean Error")
graph1.setLabels(left=('Mean Error'), bottom=('Split Ratio'))


graph2 = pg.plot(split,mean2,symbol='o',title="MONKDATA2, Mean Error")
graph2.setLabels(left=('Mean Error'), bottom=('Split Ratio'))

graph3 = pg.plot(split,mean3,symbol='o',title="MONKDATA3, Mean Error")
graph3.setLabels(left=('Mean Error'), bottom=('Split Ratio'))


#Create graph of standard deviations
graph4 = pg.plot(split,std1,symbol='o',title="MONKDATA1, Error STD")
graph4.setLabels(left=('Error Variance'), bottom=('Split Ratio'))

graph5 = pg.plot(split,std2,symbol='o',title="MONKDATA2, Error STD")
graph5.setLabels(left=('Error Variance'), bottom=('Split Ratio'))

graph6 = pg.plot(split,std3,symbol='o',title="MONKDATA3, Error STD")
graph6.setLabels(left=('Error Variance'), bottom=('Split Ratio'))


print "\nProgram finished. Exiting..."
wait = raw_input("PRESS ENTER TO TERMINATE.")
