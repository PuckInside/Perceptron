import gymnasium as gym
import keyboard
import matplotlib.pyplot as plt

from NeuralNetwork import Perceptron
import Layers as layer
import Metrics

env = gym.make("MountainCar-v0", render_mode="rgb_array") 

env.reset(seed=42)
img = env.render()

plt.ion()  
fig, ax = plt.subplots()

isClose = False
while not isClose:
    if keyboard.is_pressed('esc'):
        isClose = True
        break

    if keyboard.is_pressed('left'):
        action = 0
    elif keyboard.is_pressed('space'):
        action = 1
    elif keyboard.is_pressed('right'):
        action = 2
    else:
        action = 1


    env.step(action)
    img = env.render()

    ax.imshow(img)
    ax.axis('off')

    plt.pause(0.01)
    ax.cla()

    

env.close()
plt.ioff()