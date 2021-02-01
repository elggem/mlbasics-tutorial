# Task 1A_a
# Implement a function that prints the string "WIN" or "LOOSE" with 50% chance each.

def winOrLoose():
  if random.randint(0,1) == 1:
    print("WIN")
  else:
    print("LOOSE")

winOrLoose()
winOrLoose()
winOrLoose()
winOrLoose()
winOrLoose()


# Task 1A_b
# Implement a function that returns "WIN" or "LOOSE" with 50% chance each, 
# then use it in a loop to populate a list with ten entries from the function.

def returnWinOrLoose():
  if random.randint(0,1) == 1:
    return "WIN"
  else:
    return "LOOSE"

theList = []
for _ in range(10):
  theList.append(returnWinOrLoose())

print(theList)


# Task 1A_c
# Remove all "WIN" entries from the list generated above.

""" Why is this not the right solution?
for item in theList:
  if item == "WIN":
    theList.remove(item)

print(theList)
"""

# CORRECT:
while "WIN" in theList:
  theList.remove("WIN")

print(theList)


# Task 1B_a
# Implement a function that returns an array length 12 with random numbers.

def returnRandomArray():
  return np.random.rand(12)

returnRandomArray()


# Task 1B_b
# Use the function from 1B_a to make a 12x12 matrix of random numbers iteratively.

randomArray = []

for _ in range(12):
  randomArray.append(returnRandomArray())

randomMatrix = np.array(randomArray)

print(randomMatrix)
print(randomMatrix.shape)


# Task 1B_c
# Reshape the matrix from 1B_b to a (6,6,4) 6x6pixel RGBA image and
# give it 100% opacity and a red-tint.

stack = randomMatrix.reshape((6,6,4))

stack *= np.array([1.0,0.5,0.5,1.0])

import matplotlib.pyplot as plt
plt.imshow(stack)


# Task 1C_a
# Try to use the same model we used in the linear regression example to model the following data:

X = np.linspace(0, 2, 100)
y = 1.5 * np.sin(X**2) + np.random.randn(*X.shape) * 0.2 + 0.5

# Plot using matplotlib scatter function
plt.scatter(X, y)
plt.title("Toy Dataset 2")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Solution:

# Making a model
model = tf.keras.Sequential([
    layers.Dense(input_shape=[1,], units=1)
])

model.summary()

# Train the model
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error')
model.fit(X,y, epochs=100, verbose=0)

# Evaluate the final predictions
predictions = model.predict(X)

# Plot using matplotlib scatter function
plt.scatter(X, y)
plt.scatter(X, predictions)
plt.title("Toy Dataset")
plt.xlabel("x")
plt.ylabel("y")
plt.show()





# Task 1C_b
# Make changes to the model used in the linear regression example to model the new data:

X = np.linspace(0, 2, 100)
y = 1.5 * np.sin(X**2) + np.random.randn(*X.shape) * 0.2 + 0.5

# Plot using matplotlib scatter function
plt.scatter(X, y)
plt.title("Toy Dataset 2")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Solution:

# Making a model
model = tf.keras.Sequential([
    layers.Dense(input_shape=[1,], units=1, activation="tanh"),
    layers.Dense(units=4, activation="tanh"),
    layers.Dense(units=1)
])

model.summary()

# Train the model
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error')
history = model.fit(X,y, epochs=1000, verbose=0) ## -> 300

# Plot the loss
plt.plot(history.history['loss'])
plt.show()

# Evaluate the final predictions
predictions = model.predict(X)

# Plot using matplotlib scatter function
plt.scatter(X, y)
plt.scatter(X, predictions)
plt.title("Toy Dataset")
plt.xlabel("x")
plt.ylabel("y")
plt.show()




