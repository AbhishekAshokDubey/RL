For latest python atari games on windows: https://github.com/AbhishekAshokDubey/atari-py<br/>

This is extention of karpathy's work, with required derivation/ proof provided below.<br/>

The neural network:
![diagram](https://raw.githubusercontent.com/AbhishekAshokDubey/RL/master/ping-pong/documentation_stuff/nn_diagram.PNG)

![notations](https://raw.githubusercontent.com/AbhishekAshokDubey/RL/master/ping-pong/documentation_stuff/00_notation.png)

As always we want to maximize the probability of output (up/down action) given input (game state)


![aim](https://raw.githubusercontent.com/AbhishekAshokDubey/RL/master/ping-pong/documentation_stuff/01_aim.gif)

For ease of discussion lets drop the ![sum_sym](https://raw.githubusercontent.com/AbhishekAshokDubey/RL/master/ping-pong/documentation_stuff/02_sum.png) and, consider only one example

![eq1](https://raw.githubusercontent.com/AbhishekAshokDubey/RL/master/ping-pong/documentation_stuff/03_eq1.png)

![eq2](https://raw.githubusercontent.com/AbhishekAshokDubey/RL/master/ping-pong/documentation_stuff/04_eq2.png)

![eq3](https://raw.githubusercontent.com/AbhishekAshokDubey/RL/master/ping-pong/documentation_stuff/05_eq3.png)

![eq4](https://raw.githubusercontent.com/AbhishekAshokDubey/RL/master/ping-pong/documentation_stuff/06_eq4.png)

Note:
![relu](https://raw.githubusercontent.com/AbhishekAshokDubey/RL/master/ping-pong/documentation_stuff/07_relu.png)<br/>Since Relu is not differntiable (kink at 0), so we use sub-derivative

![eq5](https://raw.githubusercontent.com/AbhishekAshokDubey/RL/master/ping-pong/documentation_stuff/08_eq5.gif)

![eq6](https://raw.githubusercontent.com/AbhishekAshokDubey/RL/master/ping-pong/documentation_stuff/09_eq6.gif)

<br/>
<br/>

Useful tools:<br/>
https://www.codecogs.com/latex/eqneditor.php<br/>
https://go.gliffy.com/go/html5/launch<br/>
https://dillinger.io/
