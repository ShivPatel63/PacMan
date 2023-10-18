import util

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
        "*** TTU CS5368 YOUR CODE HERE ***"
        startingNode = problem.getStartState()
        if problem.isGoalState(startingNode):
            return []

        myQueue = util.Stack()
        visitedNodes = []
        # (node,actions)
        myQueue.push((startingNode, []))

        while not myQueue.isEmpty():
            currentNode, actions = myQueue.pop()
            if currentNode not in visitedNodes:
                visitedNodes.append(currentNode)

                if problem.isGoalState(currentNode):
                    return actions

                for nextNode, action, cost in problem.getSuccessors(currentNode):
                    newAction = actions + [action]
                    myQueue.push((nextNode, newAction))
        util.raiseNotDefined()

class BFS(object):
    def breadthFirstSearch(self, problem):
        "*** TTU CS5368 YOUR CODE HERE ***"
        startingNode = problem.getStartState()
        if problem.isGoalState(startingNode):
            return []

        myQueue = util.Queue()
        visitedNodes = []
        # (node,actions)
        myQueue.push((startingNode, []))

        while not myQueue.isEmpty():
            currentNode, actions = myQueue.pop()
            if currentNode not in visitedNodes:
                visitedNodes.append(currentNode)

                if problem.isGoalState(currentNode):
                    return actions

                for nextNode, action, cost in problem.getSuccessors(currentNode):
                    newAction = actions + [action]
                    myQueue.push((nextNode, newAction))           
        util.raiseNotDefined()

class UCS(object):
    def uniformCostSearch(self, problem):
        "*** TTU CS5368 YOUR CODE HERE ***"
        startingNode = problem.getStartState()
        if problem.isGoalState(startingNode):
            return []

        visitedNodes = []

        pQueue = util.PriorityQueue()
        #((coordinate/node , action to current node , cost to current node),priority)
        pQueue.push((startingNode, [], 0), 0)

        while not pQueue.isEmpty():

            currentNode, actions, prevCost = pQueue.pop()
            if currentNode not in visitedNodes:
                visitedNodes.append(currentNode)

                if problem.isGoalState(currentNode):
                    return actions

                for nextNode, action, cost in problem.getSuccessors(currentNode):
                    newAction = actions + [action]
                    priority = prevCost + cost
                    pQueue.push((nextNode, newAction, priority),priority)
        util.raiseNotDefined()
        
class aSearch (object):
    def nullHeuristic( state, problem=None):
        """
        A heuristic function estimates the cost from the current state to the nearest goal in the provided SearchProblem.  This heuristic is trivial.
        """
        return 0
    def aStarSearch(self,problem, heuristic=nullHeuristic):
        "Search the node that has the lowest combined cost and heuristic first."
        "*** TTU CS5368 YOUR CODE HERE ***"
        startingNode = problem.getStartState()
        if problem.isGoalState(startingNode):
            return []

        visitedNodes = []

        pQueue = util.PriorityQueue()
        #((coordinate/node , action to current node , cost to current node),priority)
        pQueue.push((startingNode, [], 0), 0)

        while not pQueue.isEmpty():

            currentNode, actions, prevCost = pQueue.pop()

            if currentNode not in visitedNodes:
                visitedNodes.append(currentNode)

                if problem.isGoalState(currentNode):
                    return actions

                for nextNode, action, cost in problem.getSuccessors(currentNode):
                    newAction = actions + [action]
                    newCostToNode = prevCost + cost
                    heuristicCost = newCostToNode + heuristic(nextNode,problem)
                    pQueue.push((nextNode, newAction, newCostToNode),heuristicCost)        
        util.raiseNotDefined()

