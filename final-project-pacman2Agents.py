## PACMAN FINALLY
import random
import time
from termcolor import colored

size = 8 #initialize board game 8x8
direct = "N"
score, bots = 0, 0
posx, posy = 1, 1
gx, gy = 8, 8
prev_pac, prev_ghost = "  ", "  "
action = ["U", "D", "R", "L"]
scorex, botsx = 0, 0
combo_length_pac, combo_length_gh = 0, 0
maze = []
type = ""

SOLID_COIN = "∙ "
TRANSPARENT_COIN = "⚬ "

def dots_left(state): # count the number of dots from state
    dots_count = 0
    for row in state:
        for cell in row:
            if cell == "∙ ":
                dots_count += 1
    return dots_count

def tdots_left(state): # count the number of dots from state
    tdots_count = 0
    for row in state:
        for cell in row:
            if cell == "⚬ ":
                tdots_count += 1
    return tdots_count

def game_space(): # print the game space
    print()
    print((" " * int(22 )), "『 PacMan 』")
    print((" " * int(20)), "► Pacman_Score :", round(score,2))
    print((" " * int(20)), "► Ghost_score :", round(bots,2))

    for x in range(len(maze)):
        for y in range(len(maze[x])):
            if maze[x][y] == "ᗧ ":
                print(colored(maze[x][y], "light_blue"), end="")
            elif maze[x][y] == "∙ ":
                print(colored(maze[x][y], "yellow"), end="")
            elif maze[x][y] == "⚬ ":
                print(colored(maze[x][y], "grey"), end="")
            elif maze[x][y] == "◓ ":
                print(colored(maze[x][y], "red"), end="")
            else:
                print(maze[x][y], end="")
        print()
    print()

def terminal_state(state): # boolean funciton of checking the last state which there is no solid coin left (not including transparent coins)
    return dots_left(state) == 0

def next_position(pos, action): # function that return the next position of from (action,position)
    x, y = pos
    if action == "U":
        return (x - 1, y)
    elif action == "D":
        return (x + 1, y)
    elif action == "L":
        return (x, y - 1)
    elif action == "R":
        return (x, y + 1)

def is_valid_move(maze, pos): # function that return boolean for checking valid move
    x, y = pos
    return 0 < x <= size and 0 < y <= size and maze[x][y] != "█" and maze[x][y] != "ᗧ " and maze[x][y] != "◓ "

def actions(state, character): # function that return the list of valid acitons
    valid_actions = []
    pos = (posx, posy) if character == "pacman" else (gx, gy)
    for action in ["U", "D", "L", "R"]:
        if action == "U" and is_valid_move(maze, next_position(pos, "U")):
            valid_actions.append("U")
        elif action == "D" and is_valid_move(maze, next_position(pos, "D")):
            valid_actions.append("D")
        elif action == "L" and is_valid_move(maze, next_position(pos, "L")):
            valid_actions.append("L")
        elif action == "R" and is_valid_move(maze, next_position(pos, "R")):
            valid_actions.append("R")
    return valid_actions


def result(state, action, character): # function that return the new state from the action
    # Calculate the result of applying the action to the current state
    new_state = state.copy()
    if action in actions(new_state, "pacman"):  # Pass "pacman" as the character
        if action == "U":
            move(new_state,["U"])
        elif action == "D":
            move(new_state,["D"])
        elif action == "L":
            move(new_state,["L"])
        elif action == "R":
            move(new_state,["R"])
    elif action in actions(new_state, "ghost"):  # Pass "ghost" as the character
        if action == "U":
            move2(new_state,["U"])
        elif action == "D":
            move2(new_state,["D"])
        elif action == "L":
            move2(new_state,["L"])
        elif action == "R":
            move2(new_state,["R"])

    return new_state

# Breadth First Search

def bfs(gameboard, max_depth=10): # breadth first search for finding the shortest path to the nearest point
    # problem is a tuple (maze, start, goal)
    maze, start = gameboard
    queue = [(start, [])]  # Each item in the queue is a tuple of (position, actions)
    visited = set()
    depth = 0

    while queue and depth < max_depth:
        current, actions = queue.pop(0)
        x, y = current

        if maze[x][y] == "∙ ":
            return [actions,(x,y)]

        visited.add(current)
        depth += 1

        for action in ["U", "D", "L", "R"]:
            new_pos = next_position(current, action)
            if new_pos not in visited and is_valid_move(maze, new_pos):
                queue.append((new_pos, actions + [action]))

    return None  # No path found



# Minimax Algorithm


def move(state, pacman_action): # make for pacman
    global direct, maze, posx, posy, score

    #x = input("↳ ...")
    print()
    print("Pacman's Turn :", pacman_action)
    for action in pacman_action:
        if action == "L":
            if posy - 1 > 0 and state[posx][posy - 1] != "◓ " and state[posx][posy - 1] != "█":
                state[posx][posy] = "  "
                posy -= 1
        elif action == "R":
            if posy + 1 <= size and state[posx][posy + 1] != "◓ " and state[posx][posy + 1] != "█":
                state[posx][posy] = "  "
                posy += 1
        elif action == "U":
            if posx - 1 > 0 and state[posx - 1][posy] != "◓ " and state[posx - 1][posy] != "█":
                state[posx][posy] = "  "
                posx -= 1
        elif action == "D":
            if posx + 1 <= size and state[posx + 1][posy] != "◓ " and state[posx + 1][posy] != "█":
                state[posx][posy] = "  "
                posx += 1
        
        score += scoreMinimax(state, "pacman")
        
        state[posx][posy] = "ᗧ "
        if state == maze:
            game_space()
            #time.sleep(0.25)

def move2(state, ghost_action): # move function of mainimax for ghost
    global direct, maze, gx, gy, bots

    #x = input("↳ ...")
    print()
    print("Ghost's Turn :", ghost_action)
    for action in ghost_action:
        if action == "L":
            if gy - 1 > 0 and state[gx][gy - 1] != "ᗧ " and state[gx][gy - 1] != "█":
                state[gx][gy] = "  "
                gy -= 1
                
        elif action == "R":
            if gy + 1 <= size and state[gx][gy + 1] != "ᗧ " and state[gx][gy + 1] != "█":
                state[gx][gy] = "  "
                gy += 1
                
        elif action == "U":
            if gx - 1 > 0 and state[gx - 1][gy] != "ᗧ " and state[gx - 1][gy] != "█":
                state[gx][gy] = "  "
                gx -= 1
                
        elif action == "D":
            if gx + 1 <= size and state[gx + 1][gy] != "ᗧ " and state[gx + 1][gy] != "█":
                state[gx][gy] = "  "
                gx += 1
                
        
        bots += scoreMinimax(state, "ghost")
        
        state[gx][gy] = "◓ "
        if state == maze:
            game_space()
            #time.sleep(0.25)

def minimax(state, depth, character, alpha=float('-inf'), beta=float('inf')): # return the state utility from maximizing player and minimizing player
    if character == "pacman":
        return max_value(state, depth, alpha, beta)
    else:
        return min_value(state, depth, alpha, beta)

def max_value(state, depth, alpha, beta): # max value function for maximizing player (with alpha-beta prunning)
    if terminal_state(state) or depth == 0:
        return utility(state, "pacman")
    value = float('-inf')
    for action in actions(state, "pacman"):
        value = max(value, min_value(result(state, action, "pacman"), depth - 1, alpha, beta))
        alpha = max(alpha, value)
        if alpha >= beta:  # alpha-beta pruning
            break
    return value

def min_value(state, depth, alpha, beta): # min value function for maximizing player (with alpha-beta prunning) 
    if terminal_state(state) or depth == 0:
        return utility(state, "ghost")
    value = float('inf')
    for action in actions(state, "ghost"):
        value = min(value, max_value(result(state, action, "ghost"), depth - 1, alpha, beta))
        beta = min(beta, value)
        if alpha >= beta:  # alpha-beta pruning
            break
    return value

def utility(state, character): # return the value of the state with the evaluation function
    global dots, tdots
    for x in range(len(state)):
        for y in range(len(state[x])):
            if state[x][y] == "ᗧ ":
                pacx,pacy = x,y
            if state[x][y] == "◓ ":
                ghx,ghy = x,y

    eval_score_pac = 0
    eval_score_gh = 0
    dots_eaten = dots - dots_left(state)
    tdots_eaten = tdots - tdots_left(state)
    if bfs((state,(pacx,pacy))) == None:
        npacx,npacy = pacx,pacy
    else:
        npacx,npacy = bfs((state,(pacx,pacy)))[1]
    if bfs((state,(ghx,ghy))) == None:
        nghx,nghy = ghx,ghy
    else: nghx,nghy = bfs((state,(ghx,ghy)))[1]
    nearest_dot_distance_pac = abs(pacx - npacx) + abs(pacy - npacy)
    nearest_dot_distance_gh = abs(ghx - nghx) + abs(ghy - nghy)
    # Evaluate the distances between pacman and ghost
    pacman_distance = abs(pacx - ghx) + abs(pacy - ghy)
   
    if character == "pacman":
        # Evaluation score
        eval_score_pac = 1000 * (10 - nearest_dot_distance_pac) + 500 * (10 - pacman_distance) + 100 * dots_eaten - 50 * tdots_eaten + scoreMinimax(state,"pacman") # probabilistic (the random probability for the transparent -> solid coin
        return eval_score_pac                                                                                                                                       # will be calculated in scoreMinimax() and will be considered in evaluation function of utility()
    else:                                                                                                                                                           # which affect the minimax decision-making)
        # Evaluation score
        eval_score_gh = 1000 * (10 - nearest_dot_distance_gh) + 500 * (10 - pacman_distance) + 100 * dots_eaten - 50 * tdots_eaten + scoreMinimax(state,"ghost") 
        return eval_score_gh

def scoreMinimax(state, character): # score update function of Minimax Algorithm 
    global score, bots, posx, posy, gx, gy, prev_pac, prev_ghost, combo_length_pac, combo_length_gh
    score_update = 0
    bots_update = 0

    if character == "pacman":
        if state[posx][posy] == SOLID_COIN or state[posx][posy] == TRANSPARENT_COIN:
            if state[posx][posy] == SOLID_COIN:
                if prev_pac == SOLID_COIN:
                    combo_length_pac += 1
                else:
                    combo_length_pac = 1
                    prev_pac = state[posx][posy]
            # Handle transparent coin state
            elif state[posx][posy] == TRANSPARENT_COIN:
                x = random.random()
                if x < 0.5:  # 50% chance of becoming solid again
                    state[posx][posy] = SOLID_COIN
                    print(colored("Hidden Point!", "yellow"))
                    if prev_pac == SOLID_COIN:
                        combo_length_pac += 1
                    else:
                        combo_length_pac = 1
                    prev_pac = state[posx][posy]
                else:
                    if prev_pac == SOLID_COIN:
                        score_update += combo_length_pac ** 2
                        #print(colored("Cosecutive Point!", "yellow"), bots_update)
                        print(colored("Bonus Point! :", "yellow"), combo_length_pac**2)
                    else:
                        score_update += combo_length_pac
                    combo_length_pac = 0  
        else:
            if prev_pac == SOLID_COIN:
                score_update += combo_length_pac ** 2
                print(colored("Bonus Point! :", "yellow"), combo_length_pac**2)
            else:
                score_update += combo_length_pac
            combo_length_pac = 0               
        
    
        return score_update
    else:
       if state[gx][gy] == SOLID_COIN or state[gx][gy] == TRANSPARENT_COIN:
            if state[gx][gy] == SOLID_COIN:
                if prev_ghost == SOLID_COIN:
                    combo_length_gh += 1
                else:
                    combo_length_gh = 1
                    prev_ghost = state[gx][gy]
            # Handle transparent coin state
            elif state[gx][gy] == TRANSPARENT_COIN:
                x = random.random()
                #print(x)
                if x < 0.5:  # 50% chance of becoming solid again
                    state[gx][gy] = SOLID_COIN
                    print(colored("Hidden Point!", "yellow"), end="")
                    if prev_ghost == SOLID_COIN:
                        combo_length_gh += 1
                    else:
                        combo_length_gh = 1
                    prev_ghost = state[gx][gy]
                else:
                    if prev_ghost == SOLID_COIN:
                        bots_update += combo_length_gh ** 2
                        #print(colored("Cosecutive Point!", "yellow"), bots_update)
                        print(colored("Bonus Point! :", "yellow"), combo_length_gh**2)
                    else:
                        bots_update += combo_length_gh
                    combo_length_gh = 0  

       else:
                if prev_ghost == SOLID_COIN:
                    bots_update += combo_length_gh ** 2
                    #print(colored("Cosecutive Point!", "yellow"), bots_update)
                    print(colored("Bonus Point! :", "yellow"), combo_length_gh**2)
                else:
                    bots_update += combo_length_gh
                combo_length_gh = 0  
                    
       return bots_update


# Genetic Algorithm


def moveGA(state, pacman_action): # move function of Genetic Algorithm for pacman
    #x = input("↳ ...")
    print()
    print("Pacman's Turn :", pacman_action)
    global direct, maze, posx, posy, score
    for action in pacman_action:
        if action == "L":
            if posy - 1 > 0 and state[posx][posy - 1] != "◓ " and state[posx][posy - 1] != "█":
                state[posx][posy] = "  "
                posy -= 1
                
        elif action == "R":
            if posy + 1 <= size and state[posx][posy + 1] != "◓ " and state[posx][posy + 1] != "█":
                state[posx][posy] = "  "
                posy += 1
                
        elif action == "U":
            if posx - 1 > 0 and state[posx - 1][posy] != "◓ " and state[posx - 1][posy] != "█":
                state[posx][posy] = "  "
                posx -= 1
                
        elif action == "D":
            if posx + 1 <= size and state[posx + 1][posy] != "◓ " and state[posx + 1][posy] != "█":
                state[posx][posy] = "  "
                posx += 1
                

        score += scoreGA(state, "pacman", action)
        state[posx][posy] = "ᗧ "
        if state == maze:
            game_space()

def moveGA2(state, ghost_action): # move function of Genetic Algorithm for ghost
    #x = input("↳ ...")
    print()
    print("Ghost's Turn :", ghost_action)
    global direct, maze, gx, gy, bots
    for action in ghost_action:
        if action == "L":
            if gy - 1 > 0 and state[gx][gy - 1] != "ᗧ " and state[gx][gy - 1] != "█":
                state[gx][gy] = "  "
                gy -= 1
                
        elif action == "R":
            if gy + 1 <= size and state[gx][gy + 1] != "ᗧ " and state[gx][gy + 1] != "█":
                state[gx][gy] = "  "
                gy += 1
                
        elif action == "U":
            if gx - 1 > 0 and state[gx - 1][gy] != "ᗧ " and state[gx - 1][gy] != "█":
                state[gx][gy] = "  "
                gx -= 1
                
        elif action == "D":
            if gx + 1 <= size and state[gx + 1][gy] != "ᗧ " and state[gx + 1][gy] != "█":
                state[gx][gy] = "  "
                gx += 1
                
        
        bots += scoreGA(state, "ghost", action)
        state[gx][gy] = "◓ "
        if state == maze:
            game_space()

def scoreGA(state, character, action): # score update function of  Genetic Algorithm 
    global score, bots, posx, posy, gx, gy, prev_pac, prev_ghost, combo_length_gh, combo_length_pac
    score_update = 0
    bots_update = 0
   
    if character == "pacman":
        if state[posx][posy] == SOLID_COIN or state[posx][posy] == TRANSPARENT_COIN:
            if state[posx][posy] == SOLID_COIN:
                if prev_pac == SOLID_COIN:
                    combo_length_pac += 1
                else:
                    combo_length_pac = 1
                    prev_pac = state[posx][posy]
            # Handle transparent coin state
            elif state[posx][posy] == TRANSPARENT_COIN:
                x = random.random()
                if x < 0.5:  # < 50% chance of becoming solid again
                    state[posx][posy] = SOLID_COIN
                    print(colored("Hidden Point!", "yellow"), end="")
                    if prev_pac == SOLID_COIN:
                        combo_length_pac += 1
                    else:
                        combo_length_pac = 1
                    prev_pac = state[posx][posy]
                else:
                    if prev_pac == SOLID_COIN:
                        score_update += combo_length_pac ** 2
                        #print(colored("Cosecutive Point!", "yellow"), bots_update)
                        print(colored("Bonus Point! :", "yellow"), combo_length_pac**2)
                    else:
                        score_update += combo_length_pac
                    combo_length_pac = 0  
        else:
            if prev_pac == SOLID_COIN:
                score_update += combo_length_pac ** 2
                print(colored("Bonus Point! :", "yellow"), combo_length_pac**2)
            else:
                score_update += combo_length_pac
            combo_length_pac = 0               
        
        return score_update
    else:
       if state[gx][gy] == SOLID_COIN or state[gx][gy] == TRANSPARENT_COIN:
            if state[gx][gy] == SOLID_COIN:
                if prev_ghost == SOLID_COIN:
                    combo_length_gh += 1
                else:
                    combo_length_gh = 1
                    prev_ghost = state[gx][gy]
            # Handle transparent coin state
            elif state[gx][gy] == TRANSPARENT_COIN:
                x = random.random()
                #print(x)
                if x < 0.5:  # 50% chance of becoming solid again
                    state[gx][gy] = SOLID_COIN
                    print(colored("Hidden Point!", "yellow"), end="")
                    if prev_ghost == SOLID_COIN:
                        combo_length_gh += 1
                    else:
                        combo_length_gh = 1
                    prev_ghost = state[gx][gy]
                else:
                    if prev_ghost == SOLID_COIN:
                        bots_update += combo_length_gh ** 2
                        #print(colored("Cosecutive Point!", "yellow"), bots_update)
                        print(colored("Bonus Point! :", "yellow"), combo_length_gh**2)
                    else:
                        bots_update += combo_length_gh
                    combo_length_gh = 0  
       else:
                if prev_ghost == SOLID_COIN:
                    bots_update += combo_length_gh ** 2
                    #print(colored("Cosecutive Point!", "yellow"), bots_update)
                    print(colored("Bonus Point! :", "yellow"), combo_length_gh**2)
                else:
                    bots_update += combo_length_gh
                combo_length_gh = 0  
                    
        # Additional evaluation based on distances and remaining dots
    
       return bots_update

def fitness_function_pac(state,actions): # fitness score from pacman action (focus on the combo coin)
    prev = "  "
    posx,posy = 1,1
    for x in range(len(state)):
        for y in range(len(state[x])):
            if state[x][y] == "ᗧ ":
                posx,posy = x,y
    new_state = state.copy() 
    fitness_score_pac = 0
    combo_length = 0  # Track the length of the combo
    for action in actions:
        if action == "L":
            if posy - 1 > 0 and new_state[posx][posy - 1] != "◓ " and new_state[posx][posy - 1] != "█":
                posy -= 1
                if prev == "∙ ":
                    combo_length += 1  # Increment combo length if the previous cell contained a dot
                else:
                    combo_length = 0  # Start a new combo if the previous cell did not contain a dot
                prev = new_state[posx][posy]
                
        elif action == "R":
            if posy + 1 <= size and new_state[posx][posy + 1] != "◓ " and new_state[posx][posy + 1] != "█":
                posy += 1
                if prev == "∙ ":
                    combo_length += 1
                else:
                    combo_length = 0
                prev = new_state[posx][posy]
                
        elif action == "U":
            if posx - 1 > 0 and new_state[posx - 1][posy] != "◓ " and new_state[posx - 1][posy] != "█":
                posx -= 1
                if prev == "∙ ":
                    combo_length += 1
                else:
                    combo_length = 0
                prev = new_state[posx][posy]
                
        elif action == "D":
            if posx + 1 <= size and new_state[posx + 1][posy] != "◓ " and new_state[posx + 1][posy] != "█":
                posx += 1
                if prev == "∙ ":
                    combo_length += 1
                else:
                    combo_length = 0
                prev = new_state[posx][posy]
                
        # Update score and position accordingly
        fitness_score_pac += combo_length ** 2  # Square the combo length for the score
    return fitness_score_pac

def fitness_function_ghost(state,actions): # fitness score from pacman action (focus on the combo coin)
    prev = "  "
    gx,gy = 0,0
    for x in range(len(state)):
        for y in range(len(state[x])):
            if state[x][y] == "ᗧ ":
                gx,gy = x,y
    new_state = state.copy() 
    fitness_score_ghost = 0
    combo_length = 0  # Track the length of the combo
    for action in actions:
        if action == "L":
            if gy - 1 > 0 and new_state[gx][gy - 1] != "ᗧ " and new_state[gx][gy - 1] != "█":
                gy -= 1
                if prev == "∙ ":
                    combo_length += 1  # Increment combo length if the previous cell contained a dot
                else:
                    combo_length = 0 
                prev = new_state[gx][gy]
        elif action == "R":
            if gy + 1 <= size  and new_state[gx][gy + 1] != "ᗧ " and new_state[gx][gy + 1] != "█":
                gy += 1
                if prev == "∙ ":
                    combo_length += 1 
                else:
                    combo_length = 0 
                prev = new_state[gx][gy]
        elif action == "U":
            if gx - 1 > 0 and new_state[gx - 1][gy] != "ᗧ " and new_state[gx - 1][gy]  != "█":
                gx -= 1
                if prev == "∙ ":
                    combo_length += 1  
                else:
                    combo_length = 0
                prev = new_state[gx][gy]
        elif action == "D":
            if  gx + 1 <= size  and new_state[gx + 1][gy] != "ᗧ " and new_state[gx + 1][gy] != "█":
                gx += 1
                if prev == "∙ ":
                    combo_length += 1  
                else:
                    combo_length = 0 
                prev = new_state[gx][gy]
        # Update score and position accordingly
        fitness_score_ghost += combo_length ** 2
    return fitness_score_ghost

def generate_individual(): # randomly generate the list of action with maximum 3 actions
    return [random.choice(action) for _ in range(3)]  # Generate a random sequence of actions

def crossover(parent1, parent2): # implement crossover
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual, mutation_rate=0.1): # randomly mutate the action in the list of action (individual) by 0.1 mutation rate 
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.choice(action)
    return individual

def fitness_evaluation(state,individual, character): # evaluate the state utility that depends on the fitness score of each agent 
    if character == "pacman":
        utility_pacman = fitness_function_pac(state,individual) # the utility equals to the fitness score
        return utility_pacman
    else:
        utility_ghost = fitness_function_ghost(state,individual)
        return utility_ghost 

def genetic_algorithm_pac(state): # implement the genetic algorithm for the pacman
    population_size = 10
    generations = 500
    population = [generate_individual() for _ in range(population_size)]

    for generation in range(generations):
        fitness_scores = [fitness_evaluation(state,individual,"pacman") for individual in population]
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
        fittest_individual = sorted_population[0]
        #print("Generation:", generation, "Fittest Individual:", fittest_individual, "Fitness Score:", fitness_function_pac(state,fittest_individual))
        if fitness_evaluation(state,fittest_individual, "pacman") == 5:  # Target fitness score (the most amount of combo length)

            break
        
        mating_pool = sorted_population[:int(population_size * 0.5)] # choose the mating pool from the list of sorted_population

        new_population = []                                 
        while len(new_population) < population_size:                # implement the crossover and mutation to make a new population for the next generation
            parent1, parent2 = random.sample(mating_pool, 2)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])

        population = new_population
    return fittest_individual # return a list of action that give the target fitness score

def genetic_algorithm_ghost(state): # implement the genetic algorithm for the ghost
    population_size = 10
    generations = 500
    population = [generate_individual() for _ in range(population_size)]

    for generation in range(generations):
        fitness_scores = [fitness_evaluation(state,individual,"ghost") for individual in population]
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
        fittest_individual = sorted_population[0]
        #print("Generation:", generation, "Fittest Individual:", fittest_individual, "Fitness Score:", fitness_function_ghost(state,fittest_individual))
        
        if fitness_evaluation(state,fittest_individual, "ghost") == 5:  # Target fitness score
            break
        
        mating_pool = sorted_population[:int(population_size * 0.5)]

        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(mating_pool, 2)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])

        population = new_population
    
    return fittest_individual



if __name__ == "__main__": # main funciton
    scorex_total = 0
    botsx_total = 0
    print()
    print((" " * int(6 )), "『 Final Project by Ice 🧸 』")
    print()
    print((" " * int(size / 2 - 1)), "------------ ᕦʕ •ᴥ•ʔᕤ -----------\n")
    print("Which algorithm you use? GA or Minimax: ... \n")
    type = input(colored("↳  ","yellow"))
    print((" " * int(size / 2 - 1)), "---------------------------------\n")
    for i in range(3):
        score = 0  # Reset score for each game
        bots = 0  # Reset bots for each game
        gameover = False
        maze = [[" ", "██" * size]]

        for i in range(1, size + 1):
            row = ["█"]
            for j in range(1, size + 1):
                dot_type = random.choice(["∙ ", "⚬ ", "  "])
                row.append(dot_type)
            row.append("█")
            maze.append(row)

        maze.append([" ", "██" * size])
        maze[posx][posy] = "ᗧ "
        maze[gx][gy] = "◓ "
        game_space()
        dots = dots_left(maze)
        tdots = tdots_left(maze)
        input("Press Enter To Start Game  ʕ・㉨・ʔﾉ♡ ... \n")
        if type.lower() == "minimax": # go to minimax algorithm
            while not gameover:
                
                pacman_action = []   # initialize action
           
                best_score_pac = float('-inf')  # initialize v= -inf
                for a in actions(maze, "pacman"):  # for each successor of state(maze)
                    score_pac = minimax(result(maze, a, "pacman"), 10, "pacman")  # value of max value of new_state
                    if score_pac > best_score_pac:
                        best_score_pac = score_pac  # find the most max value from the new state
                        pacman_action = a  # get the action
                if pacman_action is None:
                    pacman_action = [random.choice(["U", "D", "L", "R"])]
                
                move(maze,pacman_action)
                
                print("⊛ pallets left : ", dots_left(maze))
                
                ghost_action = []  # initialize action
                best_score_ghost = float('inf') # initialize v= inf
                for b in actions(maze, "ghost"):# for each successor of state(maze)
                    score_ghost = minimax(result(maze, b, "ghost"), 10, "ghost") # value of min value of new_state
                    if score_ghost < best_score_ghost:
                        best_score_ghost = score_ghost # find the least min value from the new state
                        ghost_action = b  
                if ghost_action is None:
                    ghost_action = [random.choice(["U", "D", "L", "R"])]
                
                move2(maze,ghost_action)
                
                print("⊛ pallets left : ", dots_left(maze))
                if terminal_state(maze):
                    gameover = True
            if score > bots:
                print("👑 Pacman Won 👑 ")
                score += 500
            elif  score < bots:
                print("👑 Ghost Won 👑 ")
                bots += 500
            else: print("Draw")
                    

            # Update total scores after each move
            scorex_total += score
            botsx_total += bots

            # Print total scores after all games are played
            print("Total Pacman Score : ", round(scorex_total,2))
            print("Total Ghost Score : ", round(botsx_total,2))
            print("Thanks!")
        
        elif type.lower() == "ga":
            while not gameover:
                
                pacman_action = []  # initialize move
          
                pacman_action = genetic_algorithm_pac(maze) # action from the GA
                if pacman_action is None:
                    pacman_action = [random.choice(["U", "D", "L", "R"])]
                
                moveGA(maze,pacman_action)
                print("⊛ pallets left : ", dots_left(maze))
                ghost_action = []  # initialize move
                ghost_action = genetic_algorithm_ghost(maze)
                if ghost_action is None:
                    ghost_action = [random.choice(["U", "D", "L", "R"])]
                moveGA2(maze,ghost_action)
                
                print("⊛ pallets left : ", dots_left(maze))
                if terminal_state(maze):
                    gameover = True
            if score > bots:
                print("👑 Pacman Won 👑 ")
                score += 500
            elif  score < bots:
                print("👑 Ghost Won 👑 ")
                bots += 500
            else: print("Draw")
                    

            # Update total scores after each move
            scorex_total += score
            botsx_total += bots

            # Print total scores after all games are played
            print("Total Pacman Score : ", round(scorex_total,2))
            print("Total Ghost Score : ", round(botsx_total,2))
            
            
        else:
            print(colored(" !!! Invalid choice. Please choose either 'GA' or 'Minimax' !!! \n", "red"))
            exit()

    if scorex_total > botsx_total:
        print("The Real Winner is ... Pacman ",colored("ᗧ ", "light_blue"))
        print("Thanks!")
    else:
        print("The Real Winner is ... Ghost ",colored("◓ ", "red"))
        print("Thanks!")

    