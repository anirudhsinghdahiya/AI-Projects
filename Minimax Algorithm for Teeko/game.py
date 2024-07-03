import random

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def make_move(self, state):
        # Use the is_drop_phase method to determine if the game is still in the drop phase
        drop_phase = self.is_drop_phase(state)

        # Use the minimax algorithm to find the best move, assuming a maximum depth of 3
        _, best_state = self.minimax(state, 3, True)  # True indicates it's AI's turn to play

        # Initialize move variables; these will hold the coordinates of the chosen move
        move = None
        for i in range(5):  # Iterate over all rows
            for j in range(5):  # Iterate over all columns
                # Check if the cell in the current state differs from the best state found
                if state[i][j] != best_state[i][j]:
                    # If the best state has AI's piece at this position, it's the move destination
                    if best_state[i][j] == self.my_piece:
                        move = (i, j)  # Set move to the new position of AI's piece
                    else:
                        # If not, it means the piece at this position was moved, set it as source
                        move_source = (i, j)  # This will be used later in non-drop phase

        # Return the move based on the game phase
        if drop_phase:
            # During the drop phase, return only the new piece's position
            return [move]
        else:
            # During the move phase, return both the source and destination of the move
            return [move, move_source]

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Evaluates the game board to determine if there's a win condition met.
        It checks for horizontal, vertical, diagonal, and 2x2 box patterns. """
        
        # Check for horizontal wins across the board
        for i in range(5):  # Loop through each row
            for j in range(2):  # Only need to start checks from the first two columns for four in a row
                # Check if four consecutive horizontal pieces are the same and not empty
                if state[i][j] != ' ' and state[i][j] == state[i][j+1] == state[i][j+2] == state[i][j+3]:
                    # Return 1 if these are the AI's pieces, else -1
                    return 1 if state[i][j] == self.my_piece else -1

        # Check for vertical wins across the board
        for i in range(5):  # Loop through each column
            for j in range(2):  # Only need to start checks from the first two rows for four in a row
                # Check if four consecutive vertical pieces are the same and not empty
                if state[j][i] != ' ' and state[j][i] == state[j+1][i] == state[j+2][i] == state[j+3][i]:
                    # Return 1 if these are the AI's pieces, else -1
                    return 1 if state[j][i] == self.my_piece else -1

        # Check for diagonal wins ("/" and "\")
        for i in range(2):  # Loop starting only from the first two rows and columns
            for j in range(2):
                # Check \ diagonal from top left to bottom right
                if state[i][j] != ' ' and state[i][j] == state[i+1][j+1] == state[i+2][j+2] == state[i+3][j+3]:
                    # Return 1 if these are the AI's pieces, else -1
                    return 1 if state[i][j] == self.my_piece else -1
                # Check / diagonal from bottom left to top right
                if state[i][j+3] != ' ' and state[i][j+3] == state[i+1][j+2] == state[i+2][j+1] == state[i+3][j]:
                    # Return 1 if these are the AI's pieces, else -1
                    return 1 if state[i][j+3] == self.my_piece else -1

        # Check for 2x2 box wins
        for i in range(4):  # Loop through the first four rows and columns
            for j in range(4):
                # Check if a 2x2 square of the same, non-empty pieces exists
                if state[i][j] != ' ' and state[i][j] == state[i][j+1] == state[i+1][j] == state[i+1][j+1]:
                    # Return 1 if these are the AI's pieces, else -1
                    return 1 if state[i][j] == self.my_piece else -1

        # If no win conditions met, return 0
        return 0  # Indicates no winner yet

    
    def succ(self, state, drop_phase):
        """ Generate all legal successor states from the current state, 
        differentiated by whether the game is in the drop or move phase. """

        successors = []  # List to store all possible successor states

        if drop_phase:
            # If it's the drop phase, add a new piece to empty spots on the board
            for i in range(5):  # Iterate over each row
                for j in range(5):  # Iterate over each column
                    if state[i][j] == ' ':  # Check if the current cell is empty
                        new_state = [row[:] for row in state]  # Create a deep copy of the current state
                        new_state[i][j] = self.my_piece  # Place the current player's piece on the empty spot
                        successors.append(new_state)  # Add the new state to the list of successors
        else:
            # If it's the move phase, consider moves of existing pieces to adjacent empty spots
            for i in range(5):  # Iterate over each row
                for j in range(5):  # Iterate over each column
                    if state[i][j] == self.my_piece:  # Check if the current cell contains AI's piece
                        # Consider all possible moves to adjacent cells
                        for di in [-1, 0, 1]:  # Offset for rows; -1, 0, 1 represent moving up, staying, and moving down
                            for dj in [-1, 0, 1]:  # Offset for columns; -1, 0, 1 represent moving left, staying, and moving right
                                ni, nj = i + di, j + dj  # Calculate new row and column indices
                                # Ensure the new indices are within the board limits and the target cell is empty
                                if 0 <= ni < 5 and 0 <= nj < 5 and state[ni][nj] == ' ':
                                    new_state = [row[:] for row in state]  # Create a deep copy of the current state
                                    new_state[i][j] = ' '  # Remove the piece from the original spot
                                    new_state[ni][nj] = self.my_piece  # Place the piece in the new spot
                                    successors.append(new_state)  # Add the new state to the list of successors

        return successors  # Return the list of all generated successor states
   
    def is_drop_phase(self, state):
        """ Determine if the game is in the drop phase based on the number of pieces on the board.
        The drop phase is active until 8 pieces have been placed on the board (4 for each player). """
        
        # Count the number of non-empty cells in the board state
        # This is achieved by iterating over each row and each cell within the row
        # and counting cells that are not empty (' ')
        pieces_count = sum(1 for row in state for cell in row if cell != ' ')
        
        # If the count of pieces on the board is less than 8, return True indicating
        # that the game is still in the drop phase. Otherwise, return False.
        return pieces_count < 8

    def minimax(self, state, depth, max_player):
        """ Perform the minimax algorithm to determine the best move from the current state.
        This function considers the current depth of recursion and whether the current move
        is to maximize or minimize the eval score.

        Args:
            state (list of list of str): The current state of the board.
            depth (int): The maximum depth of the recursion; controls lookahead moves.
            max_player (bool): True if the current move is by the maximizer; False for minimizer.

        Returns:
            tuple: A tuple containing the best evaluation score and the associated state.
        """
        
        # Check if the game is in the drop phase
        drop_phase = self.is_drop_phase(state)
        
        # Base case: if the maximum depth is reached or a terminal state is detected
        if depth == 0 or self.game_value(state) != 0:
            # Return the game value of the state and the state itself
            return self.game_value(state), state

        # If the current role is the maximizer
        if max_player:
            max_eval = float('-inf')  # Initialize the maximum evaluation to negative infinity
            best_move = None  # Initialize the best move as None
            # Generate all possible successor states
            for successor in self.succ(state, drop_phase):
                # Recursively call minimax to evaluate the successor state
                eval, _ = self.minimax(successor, depth - 1, False)
                # If the evaluation of the successor is greater than the current max_eval
                if eval > max_eval:
                    max_eval = eval  # Update max_eval
                    best_move = successor  # Update best_move
            # Return the best evaluation and the corresponding state
            return max_eval, best_move
        else:
            min_eval = float('inf')  # Initialize the minimum evaluation to infinity
            best_move = None  # Initialize the best move as None
            # Generate all possible successor states
            for successor in self.succ(state, drop_phase):
                # Recursively call minimax to evaluate the successor state
                eval, _ = self.minimax(successor, depth - 1, True)
                # If the evaluation of the successor is less than the current min_eval
                if eval < min_eval:
                    min_eval = eval  # Update min_eval
                    best_move = successor  # Update best_move
            # Return the best evaluation and the corresponding state
            return min_eval, best_move

    def heuristic_game_value(self, state):
        """ Evaluate the heuristic value of the given board state for the AI player.
        This method estimates the strategic value of the board, considering center control,
        mobility, and potential winning configurations. """

        score = 0  # Initialize the heuristic score to 0
        
        # Center control (prefer having pieces towards the center of the board)
        # Define center and near-center positions, which are typically strategically advantageous
        center_points = [(2, 2), (1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]
        for i, j in center_points:
            if state[i][j] == self.my_piece:
                score += 0.1  # Increment score for each of my pieces in center positions
        
        # Mobility (number of moves available from the current state)
        # Calculate how many legal moves the current state can yield, implying the flexibility of the AI's position
        mobility = len(self.succ(state, False))  # False assumes it's not the drop phase for mobility calculation
        score += 0.05 * mobility  # Add a small weight multiplied by the number of moves to the score

        # Potential win configurations
        # Check for situations where three pieces of AI's are aligned with an open end, indicating a potential win
        for row in range(5):
            for col in range(5):
                # Horizontal potential win check
                if col <= 1 and all(state[row][col + k] == self.my_piece for k in range(3)):
                    # Check for open ends where a win can be immediately achieved next move
                    if col > 0 and state[row][col - 1] == ' ' or col < 3 and state[row][col + 3] == ' ':
                        score += 0.3  # Significantly increase score for potential horizontal wins

                # Vertical potential win check
                if row <= 1 and all(state[row + k][col] == self.my_piece for k in range(3)):
                    # Check for open ends vertically
                    if row > 0 and state[row - 1][col] == ' ' or row < 3 and state[row + 3][col] == ' ':
                        score += 0.3  # Significantly increase score for potential vertical wins

        return score  # Return the calculated heuristic score

    
def test_win_conditions():
        player = TeekoPlayer()
        player.my_piece = 'b'
        player.opp = 'r'
        
        # Horizontal win
        test_board = [['b', 'b', 'b', 'b', ' '],
                    [' ', ' ', ' ', ' ', ' '],
                    [' ', ' ', ' ', ' ', ' '],
                    [' ', ' ', ' ', ' ', ' '],
                    [' ', ' ', ' ', ' ', ' ']]
        player.board = test_board
        assert player.game_value(test_board) == 1, "Should detect horizontal win"
        
        # Vertical win
        test_board = [['b', ' ', ' ', ' ', ' '],
                    ['b', ' ', ' ', ' ', ' '],
                    ['b', ' ', ' ', ' ', ' '],
                    ['b', ' ', ' ', ' ', ' '],
                    [' ', ' ', ' ', ' ', ' ']]
        player.board = test_board
        assert player.game_value(test_board) == 1, "Should detect vertical win"
        
        # Diagonal win
        test_board = [['b', ' ', ' ', ' ', ' '],
                    [' ', 'b', ' ', ' ', ' '],
                    [' ', ' ', 'b', ' ', ' '],
                    [' ', ' ', ' ', 'b', ' '],
                    [' ', ' ', ' ', ' ', ' ']]
        player.board = test_board
        assert player.game_value(test_board) == 1, "Should detect diagonal win"

        # 2x2 Box win
        test_board = [['b', 'b', ' ', ' ', ' '],
                    ['b', 'b', ' ', ' ', ' '],
                    [' ', ' ', ' ', ' ', ' '],
                    [' ', ' ', ' ', ' ', ' '],
                    [' ', ' ', ' ', ' ', ' ']]
        player.board = test_board
        assert player.game_value(test_board) == 1, "Should detect 2x2 box win"

        print("All tests passed.")

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")

if __name__ == "__main__":
    test_win_conditions()  
    main()
