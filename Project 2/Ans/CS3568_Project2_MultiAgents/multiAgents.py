# multiAgents.py
# --------------
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


from util import manhattanDistance, Queue
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        "*** CS3568 YOUR CODE HERE ***"
        "Decribe your function:"
        # Calculating distance from list of ghosts and foods from pacman positions and then taking reciprocals and adding that value in score
        # because closer the food higher will be the score
        #So if ghost is closer we are subtracting the score value.

        score = successorGameState.getScore()
        foodList = newFood.asList()
        for food in foodList:
            distanceToFood = manhattanDistance(newPos, food)
            if distanceToFood != 0:
              score += 1.0/distanceToFood

        for ghost in newGhostStates:
            distanceToGhost = manhattanDistance(newPos,ghost.getPosition())
            if(distanceToGhost > 0):
             score -= 1.0/distanceToGhost

        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** CS3568 YOUR CODE HERE ***"
        "PS. It is okay to define your own new functions. For example, value, min_function,max_function"

        return self.max_function(gameState=gameState, depth=0, agentIndex=0)[1]

    def is_last_node(self, gameState, depth, agentIndex):

       # this function is to check if we reached a last node in the state search tree
        if gameState.isWin():
            return gameState.isWin()
        elif gameState.isLose():
            return gameState.isLose()
        elif gameState.getLegalActions(agentIndex) == 0:
            return gameState.getLegalActions(agentIndex)
        elif depth >= self.depth * gameState.getNumAgents():
            return self.depth

    def max_function(self, gameState, depth, agentIndex):
        # Max Function Defination
        value = (float('-Inf'), None)
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            numberOfAgents = gameState.getNumAgents()
            currentPacman = (depth + 1) % numberOfAgents
            value = max([value, (self.value(gameState=successorState, depth=depth + 1, agentIndex=currentPacman), action)],
                        key=lambda idx: idx[0])
        return value

    def min_function(self, gameState, depth, agentIndex):
        # Min Function Defination
        value = (float('+Inf'), None)
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            numberOfAgents = gameState.getNumAgents()
            currentPacman = (depth + 1) % numberOfAgents
            value = min([value, (self.value(gameState=successorState, depth=depth + 1, agentIndex=currentPacman), action)],
                        key=lambda idx: idx[0])
        return value

    def value(self, gameState, depth, agentIndex):
        # Value func to identify turn of Pacman or ghosts or traverses the tree to the last node

        if self.is_last_node(gameState=gameState, depth=depth, agentIndex=agentIndex):
            return self.evaluationFunction(gameState)
        elif agentIndex is 0:
            return self.max_function(gameState=gameState, depth=depth, agentIndex=agentIndex)[0]
        else:
            return self.min_function(gameState=gameState, depth=depth, agentIndex=agentIndex)[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** CS3568 YOUR CODE HERE ***"
        "PS. It is okay to define your own new functions. For example, value, min_function,max_function"
        #Assigning alpha to -infinity and Beta to plus infinity like what we do in its algorithm
        alpha = float('-Inf')
        beta = float('+Inf')
        depth = 0
        return self.max_function(gameState=gameState, depth=depth, agentIndex=0, alpha=alpha, beta=beta)[1]

#Same implementation of functions as Minimax agent
    def is_last_node(self, gameState, depth, agentIndex):
        if gameState.isWin():
            return gameState.isWin()
        elif gameState.isLose():
            return gameState.isLose()
        elif gameState.getLegalActions(agentIndex) == 0:
            return gameState.getLegalActions(agentIndex)
        elif depth >= self.depth * gameState.getNumAgents():
            return self.depth

    def max_function(self, gameState, depth, agentIndex, alpha, beta):

        value = (float('-Inf'), None)
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            numberOfAgents = gameState.getNumAgents()
            currentPacman = (depth + 1) % numberOfAgents
            value = max([value, (
                self.value(gameState=successorState, depth=depth + 1, agentIndex=currentPacman, alpha=alpha, beta=beta),
                action)], key=lambda idx: idx[0])
            if value[0] > beta:
                return value
            alpha = max(alpha, value[0])
        return value

    def min_function(self, gameState, depth, agentIndex, alpha, beta):

        value = (float('+Inf'), None)
        legal_actions = gameState.getLegalActions(agentIndex)
        for action in legal_actions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            numberOfAgents = gameState.getNumAgents()
            currentPacman = (depth + 1) % numberOfAgents
            value = min([value, (
                self.value(gameState=successorState, depth=depth + 1, agentIndex=currentPacman, alpha=alpha, beta=beta),
                action)], key=lambda idx: idx[0])
            if value[0] < alpha:
                return value
            beta = min(beta, value[0])
        return value

    def value(self, gameState, depth, agentIndex, alpha, beta):
        if self.is_last_node(gameState=gameState, depth=depth, agentIndex=agentIndex):
            return self.evaluationFunction(gameState)
        elif agentIndex is 0:
            return self.max_function(gameState=gameState, depth=depth, agentIndex=agentIndex, alpha=alpha, beta=beta)[0]
        else:
            return self.min_function(gameState=gameState, depth=depth, agentIndex=agentIndex, alpha=alpha, beta=beta)[0]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** CS3568 YOUR CODE HERE ***"
        "PS. It is okay to define your own new functions. For example, value, min_function,max_function"
        return self.max_function(gameState=gameState, depth=0, agentIndex=0)[1]
        #util.raiseNotDefined()

    def is_last_node(self, gameState, depth, agentIndex):
        if gameState.isWin():
            return gameState.isWin()
        elif gameState.isLose():
            return gameState.isLose()
        elif gameState.getLegalActions(agentIndex) == 0:
            return gameState.getLegalActions(agentIndex)
        elif depth >= self.depth * gameState.getNumAgents():
            return self.depth

    def max_function(self, gameState, depth, agentIndex):

        value = (float('-Inf'), None)
        legalActions = gameState.getLegalActions(agentIndex)
        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            numberOfAgents = gameState.getNumAgents()
            currentPacman = (depth + 1) % numberOfAgents
            value = max([value, (
                self.value(gameState=successorState, depth=depth + 1, agentIndex=currentPacman),
                action)], key=lambda idx: idx[0])
        return value

#Instead of min we will use expectimax function in expectimax
    def expected_function(self, gameState, depth, agentIndex):
        value = list()
        legal_actions = gameState.getLegalActions(agentIndex)
        for action in legal_actions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            numberOfAgents = gameState.getNumAgents()
            current_player = (depth + 1) % numberOfAgents
            value.append(self.value(gameState=successorState, depth=depth + 1, agentIndex=current_player))
        expected_value = sum(value) / len(value)
        return expected_value

    def value(self, gameState, depth, agentIndex):

        if self.is_last_node(gameState=gameState, depth=depth, agentIndex=agentIndex):
            return self.evaluationFunction(gameState)
        elif agentIndex is 0:
            return self.max_function(gameState=gameState, depth=depth, agentIndex=agentIndex)[0]
        else:
            return self.expected_function(gameState=gameState, depth=depth, agentIndex=agentIndex)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** CS3568 YOUR CODE HERE ***"
# calculate the current game state score and  subtract
# for each feature the minimum distance
# For closest food - subtract the minimum distance. because
# larger the distance from the closest food  the larger the
# penalty. So the pacman  will prefer actions that minimize this value by moving closer to food.
# For the closest enemy ghost : Subtract the inverse of minimum
# distance multiplied by 2 factor. This means that the farther the
# pacman agent is to a non-scared ghost the less negative the score is.
# For the closest scared ghost:  subtract the minimum distance
# multiplied by 3 factor. that is  closer the pacman agent is to
# a scared ghost the less negative the score will be, enforcing
# the agent to choose action towards scared ghosts. Because by
# eating a scared ghost it will get more points.
# For the food capsules : multiply the number of remaining capsules
# by a big number so pacman should try to minimize the number of
# capsules eating them as he passes by them.


    pacmanPos = currentGameState.getPacmanPosition()
    foodPos = currentGameState.getFood().asList()
    capsulesPos = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    remainingFood = len(foodPos)
    remainingCapsules = len(capsulesPos)
    scaredGhosts = list()
    enemyGhosts = list()
    enemyGhostPos = list()
    scaredGhostsPos = list()
    score = currentGameState.getScore()

    distanceFromFood = [manhattanDistance(pacmanPos, foodPos) for foodPos in foodPos]
    if len(distanceFromFood) is not 0:
        closest_food = min(distanceFromFood)
        score -= 1.0 * closest_food

    for ghost in ghostStates:
        if ghost.scaredTimer is not 0:
            enemyGhosts.append(ghost)
        else:
            scaredGhosts.append(ghost)

    for enemy_ghost in enemyGhosts:
        enemyGhostPos.append(enemy_ghost.getPosition())

    if len(enemyGhostPos) is not 0:
        distance_from_enemy_ghost = [manhattanDistance(pacmanPos, enemy_ghost_position) for enemy_ghost_position
                                     in enemyGhostPos]
        closest_enemy_ghost = min(distance_from_enemy_ghost)
        score -= 2.0 * (1 / closest_enemy_ghost)

    for scaredGhost in scaredGhosts:
        scaredGhostsPos.append(scaredGhost.getPosition())

    if len(scaredGhostsPos) is not 0:
        distance_from_scared_ghost = [manhattanDistance(pacmanPos, scaredGhostsPos) for
                                      scaredGhostsPos in scaredGhostsPos]
        closestScaredGhost = min(distance_from_scared_ghost)
        score -= 3.0 * closestScaredGhost

    score -= 20.0 * remainingCapsules
    score -= 4.0 * remainingFood
    return score

# Abbreviation
better = betterEvaluationFunction
