# ----------------- Details -----------------------------------#
# Student Names: Khaled Elgohary , Jainil Thakkar              #
# Student Numbers: 7924188, 7884739                            #
# Course: COMP4190                                             #
# Assignment Number: 4                                         #
# -------------------------------------------------------------#

# ------------------ Code Notes ---------------------------------#
# -This code was done using sequential programming and not oop   #
# -The requested functions in the assignment is implemented      #
#  inside other methods, but not explicitly declared             #
# -All the results for each step in value iteration is printed,  #
# However this is not the case for q-learning, inorder to print  #
# each step or output from each episode simply type print(q_table)#
#----------------------------------------------------------------#

import numpy as np


def parseFile(filename):
    import ast

    # initialize empty dictionaries to store the parsed data
    data = {}
    terminal = {}
    boulder = {}

    # open the text file for reading
    with open(filename, 'r') as file:

        # loop over each line in the file
        for line in file:

            # strip any leading/trailing whitespace characters
            line = line.strip()

            # ignore any empty lines or comments
            if not line or line.startswith('#'):
                continue

            # split the line into key and value parts (with a maximum split count of 1)
            key, value = line.split('=', 1)

            # strip any leading/trailing whitespace characters from the key and value
            key = key.strip()
            value = value.strip()

            # check if the key is "Terminal" or "Boulder"
            if key == "Terminal" or key == "Boulder":

                # parse the nested values as a dictionary
                nested_dict = {}
                nested_pairs = value[1:-1].split('},')
                for pair in nested_pairs:
                    nested_key, nested_value = pair.split('={')
                    nested_key = int(nested_key)
                    nested_value = nested_value.strip('{}').split(',')
                    nested_value = tuple(map(int, nested_value))
                    nested_dict[nested_key] = nested_value

                # assign the nested dictionary to the appropriate main dictionary
                if key == "Terminal":
                    terminal = nested_dict
                else:
                    boulder = nested_dict


            # check if the key is "RobotStartState"
            elif key == "RobotStartState":

                # remove the curly braces from the value string
                value = value.strip('{}')

                # split the value string by commas and convert the resulting strings to integers
                robot_start_state = tuple(map(int, value.split(',')))

                # assign the tuple to the "RobotStartState" key in the main dictionary
                data[key] = robot_start_state

            # for all other keys, convert the value to a float and assign it to the key in the main dictionary
            else:
                data[key] = float(value)
    return data, terminal, boulder


data, terminal, boulder = parseFile('gridConf.txt')

# Arguments and useful variables below
Reward_non_terminal = -0.01
Discount = data['Discount']
MAXIMUM_ERROR = 10 ** (-3)
NUM_ACTIONS = 4  # up,down,left,right
Actions = [(1, 0), (0, -1), (-1, 0), (0, 1)]
ROWS = int(data['Vertical'])
COLS = int(data['Horizontal'])
INIT_UTILITY = [[0 for j in range(COLS)] for i in range(ROWS)]
START = data['RobotStartState']
for r in range(ROWS):
    for c in range(COLS):
        logic = False
        for key in terminal:
            if r == terminal[key][0] and c == terminal[key][1]:
                INIT_UTILITY[r][c] = terminal[key][2]
                logic = True
                break
        if (not logic):
            INIT_UTILITY[r][c] = 0

#below function creates our grid
def create_grid(grid, policy=False):
    output = ""
    entry = ""
    for r in range(ROWS):
        output += "|"
        for c in range(COLS):
            logic = True
            logic2 = True
            for key in boulder:
                if r == boulder[key][0] and c == boulder[key][1]:
                    logic = False
                    break
            for key in terminal:
                if r == terminal[key][0] and c == terminal[key][1]:
                    logic2 = False
                    entry = key
                    break
            if not logic:
                val = "WALL"
            elif not logic2:
                val = terminal[entry][2]
            else:
                if policy:
                    val = ["Down", "Left", "Up", "Right"][grid[r][c]]
                else:
                    val = str(grid[r][c])
            output += " " + str(val)[:5].ljust(5) + "|"
        output += "\n"
    print(output)

#below method is used to get the utility
def getUtility(INIT_UTILITY, r, c, action):
    ar, ac = Actions[action]
    newR, newC = r + ar, c + ac
    for key in boulder:
        if newR < 0 or newC < 0 or newR >= ROWS or newC >= COLS or ((newR, newC) in boulder[key]):
            return INIT_UTILITY[r][c]
        else:
            return INIT_UTILITY[newR][newC]

#below function calculates the utility, it accepts 3 parameters , initial utility, row, column, and action
def calculateUtility(INIT_UTILITY, r, c, action):
    utility = Reward_non_terminal
    Noise = data['Noise']
    weight = 1 - Noise
    neighbour = Noise / 2
    utility += neighbour * Discount * getUtility(INIT_UTILITY, r, c, (action - 1) % 4)
    utility += weight * Discount * getUtility(INIT_UTILITY, r, c, action)
    utility += neighbour * Discount * getUtility(INIT_UTILITY, r, c, (action + 1) % 4)
    return utility

#below is the function required to solve the value iteration, it accepts Initial utility and the depth
def solve_value_iteration(INIT_UTILITY, depth):
    print("Iterating: \n")
    counter = 0
    while True and counter < depth:
        nextUtility = [[0 for j in range(COLS)] for i in range(ROWS)]
        for r in range(ROWS):
            for c in range(COLS):
                logic = False
                for key in terminal:
                    if r == terminal[key][0] and c == terminal[key][1]:
                        nextUtility[r][c] = terminal[key][2]
                        logic = True
                        break
                if (not logic):
                    nextUtility[r][c] = 0
        error = 0
        for r in range(ROWS):
            for c in range(COLS):
                logic = False
                logic2 = False
                for key in boulder:
                    if ((r, c) in boulder[key]):
                        logic = True
                        break
                for key in boulder:
                    if ((r, c) in terminal[key]):
                        logic2 = True
                        break
                if (logic or logic2):
                    continue
                nextUtility[r][c] = max([calculateUtility(INIT_UTILITY, r, c, action) for action in range(NUM_ACTIONS)])
                error = max(error, abs(nextUtility[r][c] - INIT_UTILITY[r][c]))
        INIT_UTILITY = nextUtility
        create_grid(INIT_UTILITY)
        if error < MAXIMUM_ERROR * (1 - Discount) / Discount:
            break
        counter = counter + 1
    print("NUMBER OF ITERATIONS: " + str(counter))
    return INIT_UTILITY

#below is the function required to get optimal policy for value iteration algorithm
def getOptimalPolicy(INIT_UTILITY):
    policy = [[-1 for j in range(COLS)] for i in range(ROWS)]
    for r in range(ROWS):
        for c in range(COLS):
            logic = False
            logic2 = False
            for key in boulder:
                if r == boulder[key][0] and c == boulder[key][1]:
                    logic = True
                    break
            for key in terminal:
                if r == terminal[key][0] and c == terminal[key][1]:
                    logic2 = True
                    break
            if (logic or logic2):
                continue
            maxAction, maxUtility = None, -float("inf")
            for action in range(NUM_ACTIONS):
                u = calculateUtility(INIT_UTILITY, r, c, action)
                if u > maxUtility:
                    maxAction, maxUtility = action, u
            policy[r][c] = maxAction
    return policy

#below is the q_learning exploartion function
def solve_q_learning(Episodes):
    num_states = ROWS * COLS
    alpha = data['alpha']
    q_table = np.zeros((num_states, NUM_ACTIONS))
    best_q_table = np.zeros((ROWS * COLS,))
    for i in range(Episodes):
        state = START
        done = False
        Epsilon = 1.0
        while not done:
            state_index = state[0] * COLS + state[1]
            action = np.random.choice(NUM_ACTIONS)
            if action == 0:
                next_state = (state[0] - 1, state[1])
            elif action == 1:
                next_state = (state[0] + 1, state[1])
            elif action == 2:
                next_state = (state[0], state[1] - 1)
            elif action == 3:
                next_state = (state[0], state[1] + 1)

            if next_state[0] < 0 or next_state[0] >= ROWS or next_state[1] < 0 or next_state[1] >= COLS:
                continue

            if next_state in terminal:
                reward = terminal[next_state]
                next_state = terminal[next_state]
                done = True
            elif next_state in boulder:
                continue
            else:
                reward = -0.1

            next_state_index = next_state[0] * COLS + next_state[1]
            q_table[state_index][action] = (1 - alpha) * q_table[state_index][action] + alpha * (
                    reward + Discount * np.max(q_table[next_state_index]))
            state = next_state
            best_q_table[state_index] = np.max(q_table[state_index])
            if Epsilon < 1e-6:
                done = True
            else:
                Epsilon = Epsilon * 0.99
    policy = np.argmax(q_table, axis=1)
    return policy, q_table, best_q_table


#below method is responsible for reading in the results file used to run the required queries
def runScript(result):
    data = {}

    with open(result, "r") as file:
        for i, line in enumerate(file, start=1):
            values = line.strip().split(",")
            data[i] = {
                "state": (int(values[0]), int(values[1])),
                "episodes": int(values[2]),
                "method": values[3],
                "query": values[4]
            }

    return data

#Below function is responsible for runnning the simulation
def main():
    queriesData = runScript("result.txt")
    counter = 1
    for key in queriesData:
        state = queriesData[key]['state']
        episodes = queriesData[key]['episodes']
        method = queriesData[key]['method']
        query = queriesData[key]['query']
        INIT_UTILITY = [[0 for j in range(COLS)] for i in range(ROWS)]
        x, y = state
        for r in range(ROWS):
            for c in range(COLS):
                logic = False
                for key in terminal:
                    if r == terminal[key][0] and c == terminal[key][1]:
                        INIT_UTILITY[r][c] = terminal[key][2]
                        logic = True
                        break
                if (not logic):
                    INIT_UTILITY[r][c] = 0
        print(".................RUNNING QUERY NUMBER(" + str(counter) + ")...................")
        if method == "MDP":
            print("The initial Utility is: \n")
            create_grid(INIT_UTILITY)
            INIT_UTILITY = solve_value_iteration(INIT_UTILITY, episodes)
            policy = getOptimalPolicy(INIT_UTILITY)
            print("The optimal policy is: \n")
            create_grid(policy, True)
            if (query == "stateValue"):
                print("Result of the requested query(" + query + ")")
                print(INIT_UTILITY[x][y])
            elif (query == "bestPolicy"):
                print("Result of the requested query is already displayed")
                print("The optimal policy is already displayed as a part of our simulation")
        elif method == "RL":
            policy, q_table, best_q_table = solve_q_learning(episodes)
            print(q_table)
            output = ""
            for i in range((ROWS * COLS) - 1):
                if (i + 1) % ROWS == 0:
                    output = output + "\n"
                if policy[i] == 0:
                    output = output + "UP "
                elif policy[i] == 1:
                    output = output + "DOWN "
                elif policy[i] == 2:
                    output = output + "LEFT "
                elif policy[i] == 3:
                    output = output + "RIGHT "
            if query == "bestQValue":
                print(best_q_table)
                print("The value of bestQ_value of requested state is")
                print(best_q_table[x * COLS + ROWS])
            elif query == "bestPolicy":
                print("The best Policy is:")
                print(output)
        print("...................ENDING THE QUERY...................")
        counter += 1


main()
