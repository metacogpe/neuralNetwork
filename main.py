'''
  Welcome to my Playground! Here, you can run the Neural Network and even tinker with the parameters to discover new combinations and reach higher accuracies!

  IMPORTANT NOTE : 
    > Repl will install a few libraries need to run this program. It'll take a few    seconds only.
    > You don't need to change anything else than the CHANGABLE PARAMETERS. But you can if you wish to. Go on, work your brains out.
    > The output visuals will be automatically saved in the Files Tab on the left of the screen. You can access them from there.

  Have fun!
'''

# Importing helper functions
from network_functions import *
from helpers import *
import matplotlib as mpl

# CHANGABLE PARAMETERS
TRAINING_SAMPLES = 1000 # Total number of training samples.
LAYER_DIMS = [16, 16, 1]     # Note that the input Layer is predefined so you don't need to define it again.
EPOCHS = 2500           # Total number of Iterations.
LEARNING_RATE = 0.04    # Learning Rate to be used in Gradiend Descent.
ACTIVATION = 'relu'     # Activations used in Neural Network. Try jumping between relu/sigmoid

print(f'''
Training Samples : {TRAINING_SAMPLES}
Layer Dimensions : {LAYER_DIMS}
Epochs           : {EPOCHS}
Learning Rate    : {LEARNING_RATE}
''')

# Creating Data
X, y = create_data(TRAINING_SAMPLES, 100)
plot_data([X, y], 'Dataset')

# Initializing Random Parameters
parameters = initialize_random_parameters(LAYER_DIMS, X)

# Few logs just to keep track of our training.
cost_log, epoch_log = [], []

# Creating the Network
nn = NeuralNetwork()

# Training
print("Initializing Training...")
for epoch in range(EPOCHS):
  
  if epoch % 100 == 0 and epoch != 0:
    LEARNING_RATE -= LEARNING_RATE/10     # This is called Learning Rate Decay. It is basically done to optimize our Training.
    print("Epoch :", epoch)
  
  # Feedforwarding
  yhat, caches = nn.feedforward(X, parameters, ACTIVATION)
  # Computing and saving the logs for plotting
  cost = nn.cost(yhat, y)
  cost_log.append(cost)
  epoch_log.append(epoch+1)
  
  # Back Propagation
  grads = nn.backward_propagation(yhat, y, caches, ACTIVATION)
  # Gradient Descent
  parameters = nn.gradient_descent(parameters, grads, LEARNING_RATE)
  
  
predictions = yhat  # yhat --> the predicted output

print()
print("********** Accuracy :", accuracy_score(predictions, y), "% **********")
print("// Graphs saved. Check the files tab.")

# Saving the Cost Function Graph
plot_cost_function(epoch_log, cost_log)

# Just another way to convert predictions to 0s and 1s
yhat = np.where(predictions<0.5, 0, 1)

# Saving our Predictions graph
plot_data([X, yhat], 'Prediction')


'''
  Awesome! You just created your First Neural Network from Scratch!

  NOTE:
    > You might not be getting an amazing accuracy. That's because there are various things that we've skipped and various parameters that we haven't optimized just to not go beyond the scope of this article.

  Though, you can try to tinker with the 3 parameters:
    > TRAINING_SAMPLES
    > LAYER_DIMS
    > EPOCHS
    > LEARNING_RATE

  Lemmi hear your adventures and accuracies through your comments!
  Peace out.

'''