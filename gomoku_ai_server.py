from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import math
import random # Used for tie-breaking or fallback

app = Flask(__name__)
# Allow requests from any origin (for simple local development)
# For production, restrict this to your actual frontend's origin
CORS(app)

# --- Constants ---
SIZE = 15
PLAYER = 1
AI = 2
EMPTY = 0

# Score constants (adjust as needed for difficulty/style)
# Ensure these are sufficiently large to distinguish priorities
SCORE = {
    "FIVE": 10000000,
    "LIVE_FOUR": 1000000,
    "RUSH_FOUR": 100000,  # Includes冲四 and 活四 that become five in one step
    "LIVE_THREE": 50000,
    "SLEEP_THREE": 1000,
    "LIVE_TWO": 500,
    "SLEEP_TWO": 100,
    "LIVE_ONE": 10,
    "SLEEP_ONE": 1,
    "CENTER_BONUS": 1 # Smaller bonus for center
}

# --- Game Logic Ported to Python ---

def is_valid(board, x, y):
    return 0 <= x < SIZE and 0 <= y < SIZE

def check_win(board, player, x, y):
    if not is_valid(board, x, y) or board[y][x] != player:
        return False

    directions = [(1, 0), (0, 1), (1, 1), (1, -1)] # Horizontal, Vertical, Diag \, Diag /
    for dx, dy in directions:
        count = 1
        # Check positive direction
        for i in range(1, 5):
            nx, ny = x + i * dx, y + i * dy
            if not is_valid(board, nx, ny) or board[ny][nx] != player:
                break
            count += 1
        # Check negative direction
        for i in range(1, 5):
            nx, ny = x - i * dx, y - i * dy
            if not is_valid(board, nx, ny) or board[ny][nx] != player:
                break
            count += 1
        if count >= 5:
            return True
    return False

# --- AI Core Logic (Minimax, Evaluation) ---

class GomokuAI:
    def __init__(self, board, depth):
        self.board = board
        self.depth = depth
        self.transposition_table = {} # Optional: For memoization

    def find_best_move(self):
        start_time = time.time()
        best_score = -math.inf
        best_move = None
        immediate_win_move = None
        immediate_block_move = None

        moves = self._generate_moves(AI) # Generate moves for AI (2)

        # 1. Check for immediate AI win
        for r, c in moves:
            self.board[r][c] = AI
            if check_win(self.board, AI, c, r):
                immediate_win_move = (r, c)
                self.board[r][c] = EMPTY # Undo test move
                break
            self.board[r][c] = EMPTY # Undo test move

        if immediate_win_move:
            print(f"AI found immediate win at {immediate_win_move}")
            return {"y": immediate_win_move[0], "x": immediate_win_move[1]}

        # 2. Check for immediate player win and block
        player_moves = self._generate_moves(PLAYER) # Generate potential player moves
        block_moves = []
        for r, c in player_moves:
             # Only check if the spot is relevant (already generated as a potential move for AI is a good heuristic)
            if self.board[r][c] == EMPTY: # Check if cell is empty *before* testing
                self.board[r][c] = PLAYER
                if check_win(self.board, PLAYER, c, r):
                    block_moves.append((r, c))
                    # print(f"Player could win at ({r},{c})") # Debugging
                self.board[r][c] = EMPTY # Undo test move

        if len(block_moves) > 0:
            # If multiple block moves are required, player has a double threat (usually win)
            # Prioritize blocking. If there's only one, block it.
            # If there > 1, AI might be lost, just pick one. A more advanced heuristic could evaluate which block is better.
            immediate_block_move = block_moves[0] # Simple: block the first one found
            print(f"AI blocking player win at {immediate_block_move}")
             # Ensure the block move is actually a valid empty spot from AI's perspective
            if self.board[immediate_block_move[0]][immediate_block_move[1]] == EMPTY:
                 return {"y": immediate_block_move[0], "x": immediate_block_move[1]}
            else:
                 print(f"Warning: Block move {immediate_block_move} target is not empty. Proceeding with search.")
                 immediate_block_move = None # Reset if invalid


        # 3. Minimax search if no immediate actions
        print(f"Starting Minimax search with depth: {self.depth}")
        alpha = -math.inf
        beta = math.inf

        # Sort moves based on heuristic evaluation (optional but good for alpha-beta)
        scored_moves = []
        for r, c in moves:
            self.board[r][c] = AI
            score = self._evaluate_board(AI) # Quick evaluate
            self.board[r][c] = EMPTY
            scored_moves.append(((r, c), score))

        # Sort so higher scores are evaluated first in Max layer
        scored_moves.sort(key=lambda item: item[1], reverse=True)
        sorted_moves = [move for move, score in scored_moves]


        for r, c in sorted_moves:
            self.board[r][c] = AI
            # score = self._minimax(self.depth - 1, False, alpha, beta)
            current_hash = self._hash_board() # For transposition table
            score = self._minimax_memo(self.depth - 1, False, alpha, beta, current_hash)
            self.board[r][c] = EMPTY # Undo move

            print(f"  Move ({r},{c}) evaluated score: {score}") # Debugging output

            if score > best_score:
                best_score = score
                best_move = (r, c)
                print(f"  New best move: ({r},{c}) with score {score}")

            alpha = max(alpha, best_score)
            # No pruning at the root, we want to explore all top-level moves initially
            # if beta <= alpha:
            #     print(f"Pruning remaining moves at root") # Should not happen often at root unless depth is very low
            #     break

        if not best_move and len(moves) > 0:
            print("Minimax didn't find a better move, picking first generated move.")
            best_move = moves[0] # Fallback: pick first valid move if search fails
        elif not best_move:
            print("Error: No valid moves found!")
            # Handle case where no moves are possible (e.g., board full) - return None or error
            return None

        end_time = time.time()
        print(f"AI Calculation time: {end_time - start_time:.2f} seconds")
        print(f"AI chose move: {best_move} with score: {best_score}")
        return {"y": best_move[0], "x": best_move[1]}

    # Minimax with Alpha-Beta Pruning and Memoization
    def _minimax_memo(self, depth, is_maximizing, alpha, beta, board_hash):
        state_key = (board_hash, depth, is_maximizing)
        if state_key in self.transposition_table:
            return self.transposition_table[state_key]

        # Check for terminal state (win/loss/draw) or depth limit
        # Check win condition (optional here if checked before calling minimax)
        # Add check_win checks here for terminal states if needed for accuracy, return large scores

        if depth == 0:
            score = self._evaluate_board(AI) # Evaluate from AI's perspective
            self.transposition_table[state_key] = score
            return score

        current_player = AI if is_maximizing else PLAYER
        moves = self._generate_moves(current_player)

        if not moves: # No moves left, likely draw state if not won/lost
             self.transposition_table[state_key] = 0
             return 0

        # Optional: Sort moves within recursion too
        # scored_moves = []
        # for r, c in moves:
        #     self.board[r][c] = current_player
        #     score = self._evaluate_board(AI) # Always evaluate relative to AI Maximizer
        #     self.board[r][c] = EMPTY
        #     scored_moves.append(((r, c), score))
        # # Max player wants high scores first, Min player wants low scores first (which means high opponent scores first)
        # scored_moves.sort(key=lambda item: item[1], reverse=is_maximizing)
        # sorted_moves = [move for move, score in scored_moves]


        if is_maximizing: # AI's turn
            max_eval = -math.inf
            for r, c in moves: # Use sorted_moves if sorting enabled
                self.board[r][c] = AI
                new_hash = self._update_hash(board_hash, r, c, AI)
                eval_score = self._minimax_memo(depth - 1, False, alpha, beta, new_hash)
                self.board[r][c] = EMPTY # Undo move
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break # Beta cut-off
            self.transposition_table[state_key] = max_eval
            return max_eval
        else: # Player's turn
            min_eval = math.inf
            for r, c in moves: # Use sorted_moves if sorting enabled
                self.board[r][c] = PLAYER
                new_hash = self._update_hash(board_hash, r, c, PLAYER)
                eval_score = self._minimax_memo(depth - 1, True, alpha, beta, new_hash)
                self.board[r][c] = EMPTY # Undo move
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break # Alpha cut-off
            self.transposition_table[state_key] = min_eval
            return min_eval

    # Transposition Table Hashing (Simple Zobrist-like idea)
    # Initialize random bitstrings for each position and player state
    zobrist_table = [[[random.randint(1, 2**64 - 1) for _ in range(3)] # 0: Empty, 1: Player, 2: AI
                      for _ in range(SIZE)]
                     for _ in range(SIZE)]

    def _hash_board(self):
        h = 0
        for r in range(SIZE):
            for c in range(SIZE):
                if self.board[r][c] != EMPTY:
                    player_idx = self.board[r][c] # 1 or 2
                    h ^= self.zobrist_table[r][c][player_idx]
        return h

    def _update_hash(self, current_hash, r, c, player):
         # Assumes the move adds a piece 'player' at (r, c) which was previously EMPTY
         new_hash = current_hash
         # XOR out the empty state (could precompute or calculate on the fly if needed)
         # XOR in the new player state
         new_hash ^= self.zobrist_table[r][c][player]
         return new_hash



    # Heuristic Candidate Move Generation
    def _generate_moves(self, player_to_check_for):
        moves = set() # Use set to avoid duplicates
        has_pieces = False

        # Consider squares near existing pieces (both player and AI)
        # Radius 2 is often good
        radius = 2
        for r in range(SIZE):
            for c in range(SIZE):
                if self.board[r][c] != EMPTY:
                    has_pieces = True
                    for dr in range(-radius, radius + 1):
                        for dc in range(-radius, radius + 1):
                            if dr == 0 and dc == 0: continue
                            nr, nc = r + dr, c + dc
                            if is_valid(self.board, nc, nr) and self.board[nr][nc] == EMPTY:
                                moves.add((nr, nc))

        # If board is empty, play in center (or return center as only move)
        if not has_pieces:
            center = SIZE // 2
            if self.board[center][center] == EMPTY:
               return [(center, center)]
            else: # Should not happen if board is truly empty
                return []


        # If no neighbors found (e.g., isolated pieces), consider all empty squares? (less efficient)
        # Or just return the calculated neighbors. If moves is empty here and board isn't full, it's likely an error state.
        if not moves and has_pieces:
             print("Warning: No neighboring empty cells found, generating might be flawed or board state unusual.")
             # Fallback (less efficient): Add all empty cells if needed
             # for r in range(SIZE):
             #     for c in range(SIZE):
             #         if self.board[r][c] == EMPTY:
             #             moves.add((r, c))

        return list(moves)


    # --- Evaluation Function ---
    def _evaluate_board(self, player):
        # Evaluate board from the perspective of 'player'
        # Positive score is good for 'player', negative is bad
        ai_score = self._calculate_score_for_player(AI)
        player_score = self._calculate_score_for_player(PLAYER)

        # Return score difference, maybe weight opponent score higher for defense
        return ai_score - player_score * 1.1 # Slightly prioritize blocking opponent

    def _calculate_score_for_player(self, player):
        total_score = 0
        opponent = PLAYER if player == AI else AI

        # Evaluate rows, columns, diagonals
        lines = self._get_all_lines()
        for line in lines:
            total_score += self._evaluate_line(line, player, opponent)

        # Add positional value (optional)
        # for r in range(SIZE):
        #     for c in range(SIZE):
        #         if self.board[r][c] == player:
        #             dist_center = max(abs(r - SIZE//2), abs(c - SIZE//2))
        #             total_score += max(0, SIZE//2 - dist_center) * SCORE["CENTER_BONUS"]

        return total_score


    def _get_all_lines(self):
        lines = []
        # Rows
        for r in range(SIZE):
            lines.append(self.board[r])
        # Columns
        for c in range(SIZE):
            lines.append([self.board[r][c] for r in range(SIZE)])
        # Diagonals (top-left to bottom-right)
        for k in range(2 * SIZE - 1):
            line = []
            r_start = max(0, k - SIZE + 1)
            c_start = max(0, SIZE - 1 - k) # Corrected c_start logic? No this seem complex. Easier way:
            for i in range(SIZE):
                 r, c = r_start + i, k - (r_start + i) # Let's rethink diagonal generation
                 # Let's try another way:
                 r, c = (k-i, i) if k < SIZE else (SIZE-1-i, k-SIZE+1+i)
                 # Use a known correct diagonal generation:
                 if k < SIZE: # Top-left part including main diagonal
                     for i in range(k + 1):
                         if 0 <= k-i < SIZE and 0 <= i < SIZE: # Check bounds
                              line.append(self.board[k-i][i])
                 else: # Bottom-right part
                     for i in range(2 * SIZE - 1 - k):
                          if 0 <= SIZE-1-i < SIZE and 0<= k-SIZE+1+i < SIZE: # Check bounds
                             line.append(self.board[SIZE-1-i][k-SIZE+1+i])
            if len(line) >= 5: lines.append(line)

        # Diagonals (top-right to bottom-left) - Antidiagonals
        for k in range(2 * SIZE - 1):
            line = []
            for i in range(SIZE):
                 r, c = i, k-i
                 if 0 <= r < SIZE and 0 <= c < SIZE: # Check bounds are within board
                     line.append(self.board[r][c])
            if len(line) >= 5: lines.append(line) # This is slightly different from previous JS, check carefully

        # --- Alternative simpler diagonal grabbing ---
        lines_alt = []
         # Rows
        for r in range(SIZE):
            lines_alt.append(self.board[r])
        # Columns
        for c in range(SIZE):
            lines_alt.append([self.board[r][c] for r in range(SIZE)])
        # Diagonals (TL to BR)
        for i in range(-(SIZE - 1), SIZE):
            line = []
            for j in range(SIZE):
                r, c = j, j + i
                if 0 <= r < SIZE and 0 <= c < SIZE:
                    line.append(self.board[r][c])
            if len(line) >= 5: lines_alt.append(line)
        # Anti-diagonals (TR to BL)
        for i in range(2 * SIZE - 1):
            line = []
            for j in range(SIZE):
                r, c = j, i - j
                if 0 <= r < SIZE and 0 <= c < SIZE:
                    line.append(self.board[r][c])
            if len(line) >= 5: lines_alt.append(line)

        # return lines # Use original if confident, otherwise use lines_alt
        return lines_alt



    def _evaluate_line(self, line, player, opponent):
        score = 0
        n = len(line)

        for i in range(n):
            # Evaluate patterns centered at i
            # Find consecutive pieces for 'player' starting from i
            count = 0
            if line[i] == player:
                count = 1
                # Look right
                for j in range(i + 1, n):
                    if line[j] == player:
                        count += 1
                    else:
                        break
                # Look left (no need, will be covered when i reaches those points)

                # Now evaluate the 'count'-long sequence found
                if count >= 5:
                    score += SCORE["FIVE"]
                    continue # No need to check smaller patterns if five-in-a-row

                # Check ends for openness
                left_empty = (i > 0 and line[i - 1] == EMPTY)
                right_empty = (i + count < n and line[i + count] == EMPTY)
                left_opponent = (i > 0 and line[i - 1] == opponent) or i == 0 # Treat border as opponent
                right_opponent = (i + count < n and line[i + count] == opponent) or i + count == n

                # --- Pattern Matching ---
                if count == 4:
                    if left_empty and right_empty: score += SCORE["LIVE_FOUR"]
                    elif left_empty or right_empty: score += SCORE["RUSH_FOUR"]
                    # else: dead four (no score or very low)

                elif count == 3:
                     # Check for OO_XXX_O pattern (potential live four)
                     if left_empty and (i>1 and line[i-2] == EMPTY): # O_XXX... check O
                         # potential O P P P _ ?
                         if right_empty: # O P P P O ?
                             pass # This is live three, handled below
                     if right_empty and (i+count+1<n and line[i+count+1] == EMPTY): # ...XXX_O O
                         # potential ? _ P P P O
                          if left_empty: # ? O P P P O
                              pass # This is live three

                     # Check for simple live/sleep three
                     if left_empty and right_empty:
                         # Check for edge cases blocking live three like OXXXOO or OOXXXO
                         is_blocked_live_three = False
                         if i > 0 and line[i-1] == opponent: is_blocked_live_three=True
                         if i+count < n and line[i+count] == opponent: is_blocked_live_three=True
                         # More checks needed OXXXO_ or _OXXXO
                         if not ( (i>0 and line[i-1]==opponent) or (i+count<n and line[i+count]==opponent) ):
                              score += SCORE["LIVE_THREE"]
                         else: # It's actually a sleepy three if one side blocked by opponent
                              score += SCORE["SLEEP_THREE"]

                     elif left_empty or right_empty:
                         # Needs careful check for opponent immediate block
                         is_truly_sleepy = True
                         # Example O P P P X or X P P P O
                         if left_empty and right_opponent: score += SCORE["SLEEP_THREE"]
                         elif right_empty and left_opponent: score += SCORE["SLEEP_THREE"]
                         # Need to consider gaps like P_PPP or PP_PP -> handled by iterating i

                     # else: dead three (no score or very low)

                elif count == 2:
                     if left_empty and right_empty:
                         # Check for open gap _OXXO_
                         if (i+count+1 < n and line[i+count+1] == EMPTY) or \
                            (i > 1 and line[i-2] == EMPTY):
                             score += SCORE["LIVE_TWO"] #Potential for THREE
                         else: # Simple OXXO
                             score += SCORE["LIVE_TWO"] / 2 # Less valuable? Or use SLEEP_TWO?
                     elif left_empty or right_empty:
                        # Check context OXX_ or _XXO
                         is_truly_sleepy = False
                         if left_empty and (i+count < n and line[i+count] == EMPTY): # OXX__
                             score += SCORE["SLEEP_TWO"]
                         elif right_empty and (i>0 and line[i-1] == EMPTY): # __XXO
                             score += SCORE["SLEEP_TWO"]
                         elif left_empty and right_opponent: # OXX X
                              score += SCORE["SLEEP_TWO"]
                         elif right_empty and left_opponent: # X XXO
                              score += SCORE["SLEEP_TWO"]

                elif count == 1:
                     if left_empty and right_empty:
                         # Check context O X O
                         if (i>1 and line[i-2]==EMPTY) and (i+count+1 < n and line[i+count+1]==EMPTY): # OO X OO
                              score += SCORE["LIVE_ONE"]
                         else:
                              score += SCORE["LIVE_ONE"] / 2
                     elif left_empty or right_empty:
                          score += SCORE["SLEEP_ONE"]

            elif line[i] == EMPTY:
                # Check for gapped patterns like X_XX or XX_X (potential rush fours etc)
                # This complicates evaluation significantly. Sticking to contiguous checks for now.
                # A more robust evaluator would use string matching or dedicated pattern checks.
                pass

        # --- Advanced Pattern Matching (Simplified Example) ---
        # This requires a different approach than the simple loop above
        line_str = "".join(map(str, line)) # Convert line to string '0', '1', '2'
        p_str = str(player)
        o_str = str(opponent)
        e_str = str(EMPTY)

        # Find patterns using string methods or regex (can be slow)
        # Example patterns for player 'p'
        patterns_scores = {
            # Five in a row
            f"{p_str*5}": SCORE["FIVE"] * 10, # Give HUGE bonus if already won in line
            # Live four: OXXXXO
            f"{e_str}{p_str*4}{e_str}": SCORE["LIVE_FOUR"],
            # Rush four: XOOOO_ or _OOOOX or OO_OO or O_OOO etc.
            f"{o_str}{p_str*4}{e_str}": SCORE["RUSH_FOUR"],
            f"{e_str}{p_str*4}{o_str}": SCORE["RUSH_FOUR"],
             f"{p_str}{p_str}{e_str}{p_str}{p_str}": SCORE["RUSH_FOUR"] * 0.9, # slightly less urgent?
             f"{p_str}{e_str}{p_str}{p_str}{p_str}": SCORE["RUSH_FOUR"] * 0.9,
             f"{p_str*3}{e_str}{p_str}": SCORE["RUSH_FOUR"] * 0.9,
            # Live three: OXXXO
            f"{e_str}{p_str*3}{e_str}": SCORE["LIVE_THREE"],
            # Sleepy three variants: XOOO_, _OOOX, XO_OO, XOO_O etc.
            f"{o_str}{p_str*3}{e_str}": SCORE["SLEEP_THREE"],
            f"{e_str}{p_str*3}{o_str}": SCORE["SLEEP_THREE"],
            f"{o_str}{p_str}{e_str}{p_str}{p_str}{e_str}": SCORE["SLEEP_THREE"] * 0.8, # Gapped
            f"{e_str}{p_str*2}{e_str}{p_str}{o_str}": SCORE["SLEEP_THREE"] * 0.8,
            # Live two: OXXO
            f"{e_str}{e_str}{p_str*2}{e_str}{e_str}": SCORE["LIVE_TWO"], # OOXXOO
            f"{e_str}{p_str}{e_str}{p_str}{e_str}": SCORE["LIVE_TWO"], # O X O X O
            f"{e_str}{p_str*2}{e_str}{e_str}": SCORE["LIVE_TWO"] * 0.8, # OXXOO
            f"{e_str}{e_str}{p_str*2}{e_str}": SCORE["LIVE_TWO"] * 0.8, # OOXXO

            # Sleepy two: XOO__, __OOX, XO_O_, etc.
            f"{o_str}{p_str*2}{e_str}{e_str}": SCORE["SLEEP_TWO"],
            f"{e_str}{e_str}{p_str*2}{o_str}": SCORE["SLEEP_TWO"],
            f"{o_str}{p_str}{e_str}{p_str}{e_str}": SCORE["SLEEP_TWO"],
            f"{e_str}{p_str}{e_str}{p_str}{o_str}": SCORE["SLEEP_TWO"],
        }

        pattern_score = 0
        for pattern, value in patterns_scores.items():
             if pattern in line_str:
                 pattern_score += value
                 # Be careful: patterns might overlap (e.g., OOOOO contains OOOO).
                 # A more robust system counts non-overlapping occurrences or uses weights.
                 # For simplicity, just summing might be okay as stronger patterns have much higher scores.

        # Return the sum of contiguous score and pattern score? Or just one?
        # Pattern score is potentially more accurate if patterns are defined well.
        # Let's use pattern_score primarily, but keep 'score' from contiguous check?
        # return max(score, pattern_score) # Or combine them? Test what works best.
        return pattern_score + score * 0.1 # Give pattern matching higher weight


# --- Flask Route ---

@app.route('/ai_move', methods=['POST'])
def get_ai_move():
    data = request.json
    if not data or 'board' not in data or 'depth' not in data:
        return jsonify({"error": "Missing board or depth in request"}), 400

    board_state = data['board']
    search_depth = int(data['depth'])

    # Basic validation
    if not isinstance(board_state, list) or len(board_state) != SIZE or \
       not all(isinstance(row, list) and len(row) == SIZE for row in board_state):
        return jsonify({"error": "Invalid board format"}), 400

    if not isinstance(search_depth, int) or search_depth <= 0:
        return jsonify({"error": "Invalid depth"}), 400

    print(f"\nReceived request: Depth={search_depth}")
    # print("Board state:") # Optional: print board for debugging
    # for row in board_state:
    #     print(row)

    ai = GomokuAI(board_state, search_depth)
    best_move = ai.find_best_move()

    if best_move:
        return jsonify({"move": best_move})
    else:
        # This should ideally not happen unless the board is full and it's AI's turn (draw?)
        # Or if generate_moves failed completely
        print("Error: AI could not determine a move.")
        # Find *any* empty spot as a last resort if possible
        for r in range(SIZE):
            for c in range(SIZE):
                if board_state[r][c] == EMPTY:
                    print(f"Returning fallback empty spot: ({r},{c})")
                    return jsonify({"move": {"y": r, "x": c}})
        # If no empty spots, it's a draw or error
        return jsonify({"error": "AI failed to find a move (no empty spots or error)"}), 500


if __name__ == '__main__':
    # Run on 0.0.0.0 to be accessible from other devices on the network if needed
    # Use debug=True for development (auto-reloads), False for production
    app.run(host='0.0.0.0', port=5000, debug=True)
