# Deep Q Network
## Paper background
This algorithm comes from the revolutionary DeepMind Paper, [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)
Although the basis of the ideas behind the paper are not new, there are several key innovations in the paper that allowed the authors
to successfully use a neural network to successfully learng an environment like an Atari game.

## Implementation details
```
Agent 
    act(s): choose an actions based on a state. Uses eps-greedy exploitation and thus as a eps probability of choosing a random action
    observe(sars_): pass in environment information in the form of current state, action, reward, and future state (s_) after action is taken
    replay(): sample from the memory and train the network using new values you calculate
Brain 
    predict(state): approximates the value of every action given a specific state 
    train(states, values) : trains the network with the states and value of those states
Environment
    run(agent): runs the agent in the environment
Memory
    add(sars_): adds a sample to the end of the memory
    sample(size): draws a random sample w/o replacement from the memory
```
