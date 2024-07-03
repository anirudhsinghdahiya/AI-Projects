
import heapq

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    distance = 0
    index_to_xy = lambda index: (index % 3, index // 3)
    goal_indices = {tile: idx for idx, tile in enumerate(to_state) if tile != 0}

    for idx, tile in enumerate(from_state):
        if tile != 0:
            if tile in goal_indices:
                goal_idx = goal_indices[tile]
                x1, y1 = index_to_xy(idx)
                x2, y2 = index_to_xy(goal_idx)
                distance += abs(x1 - x2) + abs(y1 - y2)
            else:
                raise ValueError(f"Unexpected tile value {tile} found, which is not in the goal state.")

    return distance



def print_succ(state):
    """
    Prints the list of all the valid successors in the puzzle along with their Manhattan distances.

    INPUT: 
        A state (list of length 9)
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))



def get_succ(state):
    """
    Generate all valid successors for a given state of the puzzle.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle, sorted by their numerical representation.
    """
    succ_states = []
    index_to_xy = lambda index: (index % 3, index // 3)  # Converts index to x, y coordinates on a 3x3 grid
    xy_to_index = lambda x, y: y * 3 + x               # Converts x, y coordinates back to list index

    # Find indices of empty slots (represented by '0')
    empty_indices = [i for i, x in enumerate(state) if x == 0]

    # Possible movements: left, right, up, down
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for empty_index in empty_indices:
        x, y = index_to_xy(empty_index)

        for move in moves:
            new_x, new_y = x + move[0], y + move[1]

            # Check if new coordinates are within the grid boundaries
            if 0 <= new_x < 3 and 0 <= new_y < 3:
                new_index = xy_to_index(new_x, new_y)
                new_state = state[:]
                # Swap the empty tile with the adjacent tile
                new_state[empty_index], new_state[new_index] = new_state[new_index], new_state[empty_index]
                succ_states.append(new_state)

    # Sort the states treating each as a nine-digit integer
    return sorted(succ_states, key=lambda x: tuple(x))

def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    # Priority queue initialized with the starting state
    pq = []
    heapq.heappush(pq, (get_manhattan_distance(state), state, 0, -1))  # (priority, state, g, parent_index)

    # This will store information for each state: (state, g, h, parent_index)
    state_info = {tuple(state): (0, get_manhattan_distance(state), -1)}

    max_length = 0
    state_info_list = []

    while pq:
        current_priority, current_state, current_g, parent_index = heapq.heappop(pq)
        current_h = current_priority - current_g

        # Append to state info list for output later
        state_info_list.append((current_state, current_h, current_g))
        max_length = max(max_length, len(pq))  # Update max queue length

        # If the goal state is reached, break and reconstruct the path
        if current_state == goal_state:
            break

        # Explore successors
        for successor in get_succ(current_state):
            g = current_g + 1  # Each move costs 1
            h = get_manhattan_distance(successor)
            f = g + h

            if tuple(successor) not in state_info or g < state_info[tuple(successor)][0]:
                # Update the state information with the new lower cost path
                state_info[tuple(successor)] = (g, h, len(state_info_list) - 1)
                heapq.heappush(pq, (f, successor, g, len(state_info_list) - 1))

    # Print the path from the start state to the goal state
    current_index = len(state_info_list) - 1
    final_path = []

    while current_index != -1:
        state_info_entry = state_info_list[current_index]
        final_path.append(state_info_entry)
        current_index = state_info[tuple(state_info_entry[0])][2]  # Get the parent index

    # The path is collected in reverse order, so reverse it
    for state_info in reversed(final_path):
        current_state, h, move = state_info
        print(current_state, "h={}".format(h), "moves: {}".format(move))
    print("Max queue length: {}".format(max_length))

if __name__ == "__main__":
    print("Testing with the initial state one move away from the goal:")
    solve([1, 2, 3, 4, 5, 6, 7, 0, 0])
    print()

    print("Testing with a simple case requiring a few moves:")
    solve([1, 2, 3, 4, 5, 6, 0, 7, 0])
    print()

    print("Testing with a complex state:")
    solve([2, 5, 1, 4, 0, 6, 7, 0, 3])
    print()

    print("Testing with another random state, using only valid tiles:")
    solve([3, 1, 2, 6, 4, 5, 0, 7, 0])  # Ensure this uses only valid tile numbers
    print()