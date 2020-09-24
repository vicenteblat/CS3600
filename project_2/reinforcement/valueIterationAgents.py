# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()      # A Counter is a dict with default 0
        self.valuesNext = util.Counter()  # Counter of values (utilities) for next iteration
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # Initialize utilities for every state in S to 0.0, except for the terminal state
        # which is initialized to its reward
        # VALUES ARE INITIALIZED WHEN COUNTERS ARE INITIALIZED

        # Repeat process for given number of iterations
        while self.iterations > 0:
            # For each state, update its utility using the Bellman eqn
            for state in self.mdp.getStates():
                sumList = []
                for action in self.mdp.getPossibleActions(state):
                    actionSum = 0
                    for possibleAction in self.mdp.getTransitionStatesAndProbs(state, action):
                        nextState = possibleAction[0]
                        nextStateUtility = self.values[nextState]
                        probability = possibleAction[1]
                        actionSum += probability * nextStateUtility
                    sumList.append(actionSum)
                if self.mdp.isTerminal(state):
                    self.valuesNext[state] = self.mdp.getReward(state, None, None)
                else:
                    self.valuesNext[state] = self.mdp.getReward(state, None, None) + self.discount * max(sumList)
            self.values = self.valuesNext
            self.iterations -= 1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        else:
            bestAction = None
            highestValue = 0
            nextState = None
            for action in self.mdp.getPossibleActions(state):
                if action == 'north':
                    nextState = (state[0], state[1] + 1)
                elif action == 'south':
                    nextState = (state[0], state[1] - 1)
                elif action == 'east':
                    nextState = (state[0] + 1, state[1])
                elif action == 'west':
                    nextState = (state[0] - 1, state[1])

                nextStateValue = self.values[nextState]
                if nextStateValue > highestValue:
                    highestValue = nextStateValue
                    bestAction = action
            return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
