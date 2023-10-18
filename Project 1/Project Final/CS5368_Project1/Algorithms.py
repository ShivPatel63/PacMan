import util
from queue import PriorityQueue

from search import nullHeuristic


class DFS(object):
    def depthFirstSearch(self, problem):
        """
        Search the deepest nodes in the search tree first
        [2nd Edition: p 75, 3rd Edition: p 87]

        Your search algorithm needs to return a list of actions that reaches
        the goal.  Make sure to implement a graph search algorithm
        [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

        To get started, you might want to try some of these simple commands to
        understand the search problem that is being passed in:

        print "Start:", problem.getStartState()
        print "Is the start a goal?", problem.isGoalState(problem.getStartState())
        print "Start's successors:", problem.getSuccessors(problem.getStartState())
        """
        "*** TTU CS3568 YOUR CODE HERE ***"
        #initializing the variables
        currentState = problem.getStartState()
        AlreadyVisitedStatesList = []
        statesStack = util.Stack()
        statesStack.push((currentState, []))

        # Extract nodes and directions from each state and check for goal state
        while not statesStack.isEmpty():
         node, path = statesStack.pop()
         #Checking If current node is goal, return the path
         if problem.isGoalState(node):
             return path
         #adding current node in already visited node list
         AlreadyVisitedStatesList.append(node)
         #check for successors from current state
         successorStates = problem.getSuccessors(node)
         # iterate through all successors and check it they exists in visited states list
         #If not exist, assigned that state to current and push directions in stack
         for state in successorStates:
                  if not state[0] in AlreadyVisitedStatesList:
                     tracePath = path + [state[1]]
                     statesStack.push((state[0],tracePath))

        return tracePath
        util.raiseNotDefined()


class BFS(object):
    def breadthFirstSearch(self, problem):
        "*** TTU CS3568 YOUR CODE HERE ***"
        currentState = problem.getStartState()
        AlreadyVisitedStatesList = [currentState]
        statesQueue = util.Queue()
        statesQueue.push((currentState, []))

        # Extract nodes and directions from each state and check for goal state
        while not statesQueue.isEmpty():
            node, path = statesQueue.pop()
            if problem.isGoalState(node):
                return path

            # checking for successors from current state
            successorStates = problem.getSuccessors(node)
            # iterate through all successors and check it they exists in already visited states list
            # If not exist, assigned that state to current and push directions in stack
            for state in successorStates:
                if not state[0] in AlreadyVisitedStatesList:
                    AlreadyVisitedStatesList.append(state[0])
                    tracePath = path + [state[1]]
                    statesQueue.push((state[0], tracePath))
        return path
        util.raiseNotDefined()

class UCS(object):
    def uniformCostSearch(self, problem):
        "*** TTU CS3568 YOUR CODE HERE ***"
        currentState = problem.getStartState()
        AlreadyVisitedStatesList = []
        statesQueue = util.PriorityQueue()
        statesQueue.push((currentState, []),0)
        # Extract nodes and directions from each state and check for goal state
        while not statesQueue.isEmpty():
            node, path = statesQueue.pop()
            if problem.isGoalState(node):
                return path

            # checking for successors from current state
            if node not in AlreadyVisitedStatesList:
                successorStates = problem.getSuccessors(node)
            # iterate through all successors and check it they exists in already visited states list
            # If not exist, assigned that state to current and push directions in Q
                for state in successorStates:
                    if state[0] not in AlreadyVisitedStatesList:
                        tracePath = path + [state[1]]
                        statesQueue.push((state[0], tracePath),problem.getCostOfActions(tracePath))

            AlreadyVisitedStatesList.append(node)
        return path
        util.raiseNotDefined()
        
class aSearch (object):
    def nullHeuristic( state, problem=None):
        """
        A heuristic function estimates the cost from the current state to the nearest goal in the provided SearchProblem.  This heuristic is trivial.
        """
        return 0
    def aStarSearch(self,problem, heuristic=nullHeuristic):
        "Search the node that has the lowest combined cost and heuristic first."
        "*** TTU CS3568 YOUR CODE HERE ***"
        currentState = problem.getStartState()
        AlreadyVisitedStatesList = []
        statesQueue = util.PriorityQueue()
        statesQueue.push((currentState,[]), nullHeuristic(currentState,problem))
        while not statesQueue.isEmpty():
            node, path = statesQueue.pop()
            if problem.isGoalState(node):
                return path
            # checking for successors from current state
            if node not in AlreadyVisitedStatesList:
                successorStates = problem.getSuccessors(node)
            # iterate through all successors and check it they exists in already visited states list
            # If not exist, assigned that state to current and push directions in stack
                for state in successorStates:
                    if state[0] not in AlreadyVisitedStatesList:
                        tracePath = path + [state[1]]
                        totalCost = problem.getCostOfActions(tracePath) + heuristic(state[0], problem)
                        statesQueue.push((state[0], tracePath), totalCost)
            # add current node in visited node list
            AlreadyVisitedStatesList.append(node)
        return path
        util.raiseNotDefined()

