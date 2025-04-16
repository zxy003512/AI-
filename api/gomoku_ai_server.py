from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import math
import random
import copy # Needed for deep copying board state for threads
import threading # Needed for locking
import concurrent.futures # Needed for thread pool

app = Flask(__name__)
# Configure CORS for Vercel deployment (allow requests from your Vercel domain)
# For simplicity during development and deployment, allowing all origins is common,
# but restrict this in a real production scenario if needed.
# CORS(app, resources={r"/ai_move": {"origins": "YOUR_VERCEL_APP_URL or *"}})
CORS(app) # Allows all origins by default

# --- Constants ---
SIZE = 15
PLAYER = 1
AI = 2
EMPTY = 0

# Score constants (adjust as needed)
SCORE = {
    "FIVE": 10000000,
    "LIVE_FOUR": 1000000,
    "RUSH_FOUR": 100000,
    "LIVE_THREE": 50000,
    "SLEEP_THREE": 1000,
    "LIVE_TWO": 500,
    "SLEEP_TWO": 100,
    "LIVE_ONE": 10,
    "SLEEP_ONE": 1,
    "CENTER_BONUS": 1
}

# --- Game Logic ---
def is_valid(board, x, y):
    return 0 <= x < SIZE and 0 <= y < SIZE

def check_win(board, player, x, y):
    # (Keep the check_win function exactly as it was in the original file)
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

# --- AI Core Logic ---
class GomokuAI:
    # Transposition Table Hashing (Zobrist) - Placed outside __init__ to be class-level
    zobrist_table = [[[random.randint(1, 2**64 - 1) for _ in range(3)]
                      for _ in range(SIZE)]
                     for _ in range(SIZE)]
    tt_lock = threading.Lock() # Lock for thread-safe access to transposition table

    def __init__(self, board, depth):
        self.initial_board = board # Store the initial board state
        self.depth = depth
        self.transposition_table = {} # Instance-specific TT (could be shared if needed carefully)

    def _hash_board(self, board_state):
        # Calculate hash based on the provided board_state
        h = 0
        for r in range(SIZE):
            for c in range(SIZE):
                if board_state[r][c] != EMPTY:
                    player_idx = board_state[r][c]
                    h ^= self.zobrist_table[r][c][player_idx]
        return h

    def _update_hash(self, current_hash, r, c, player):
        # Update hash based on a move applied
        new_hash = current_hash
        new_hash ^= self.zobrist_table[r][c][player] # XOR in the new piece
        # Note: If the cell wasn't empty before (e.g., undoing), XOR out the old piece first
        # new_hash ^= self.zobrist_table[r][c][EMPTY] # XOR out empty (if needed)
        return new_hash

    def _get_board_for_thread(self):
        # Creates a deep copy of the initial board for thread safety
        return copy.deepcopy(self.initial_board)

    def find_best_move(self):
        start_time = time.time()
        best_score = -math.inf
        best_move = None
        immediate_win_move = None
        immediate_block_move = None

        current_board = self._get_board_for_thread() # Use a copy for initial checks

        # --- 1. Check for immediate AI win ---
        moves_for_ai = self._generate_moves(current_board, AI)
        for r, c in moves_for_ai:
            current_board[r][c] = AI
            if check_win(current_board, AI, c, r):
                immediate_win_move = (r, c)
                # No need to undo, current_board is a temporary copy for this check phase
                break
            current_board[r][c] = EMPTY # Undo for the next check in this loop

        if immediate_win_move:
            print(f"AI found immediate win at {immediate_win_move}")
            return {"y": immediate_win_move[0], "x": immediate_win_move[1]}

        # --- 2. Check for immediate player win and block ---
        current_board = self._get_board_for_thread() # Get a fresh copy
        moves_for_player = self._generate_moves(current_board, PLAYER)
        block_moves = []
        for r, c in moves_for_player:
            if current_board[r][c] == EMPTY:
                current_board[r][c] = PLAYER
                if check_win(current_board, PLAYER, c, r):
                    block_moves.append((r, c))
                current_board[r][c] = EMPTY # Undo test move

        if len(block_moves) > 0:
            immediate_block_move = block_moves[0] # Simple: block the first one
            print(f"AI blocking player win at {immediate_block_move}")
            # Ensure the block move is valid on the *original* board state
            if self.initial_board[immediate_block_move[0]][immediate_block_move[1]] == EMPTY:
                 return {"y": immediate_block_move[0], "x": immediate_block_move[1]}
            else:
                 print(f"Warning: Block move {immediate_block_move} target not empty on original board. Proceeding.")
                 immediate_block_move = None


        # --- 3. Multi-threaded Minimax Search ---
        print(f"Starting Multi-threaded Minimax search with depth: {self.depth}")
        alpha = -math.inf
        beta = math.inf

        # Generate candidate moves on the original board state
        candidate_moves = self._generate_moves(self.initial_board, AI)

        if not candidate_moves:
             print("Error: No valid moves found for AI!")
             # Handle no moves case (maybe return first empty?) - Added fallback later
             return None

        # Optional: Sort moves heuristically before parallel processing
        scored_moves = []
        temp_board_sort = self._get_board_for_thread()
        for r, c in candidate_moves:
            temp_board_sort[r][c] = AI
            score = self._evaluate_board(temp_board_sort, AI) # Quick evaluate on temp board
            temp_board_sort[r][c] = EMPTY
            scored_moves.append(((r, c), score))
        scored_moves.sort(key=lambda item: item[1], reverse=True)
        sorted_moves = [move for move, score in scored_moves]

        # Use ThreadPoolExecutor to evaluate top-level moves in parallel
        move_results = {}
        # Adjust max_workers based on CPU cores and desired parallelism. Too many can hurt performance.
        # Using None often defaults to the number of cores.
        with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
            future_to_move = {}
            for r, c in sorted_moves:
                # Create a DEEP COPY of the board for this thread/task
                board_copy = self._get_board_for_thread()
                board_copy[r][c] = AI # Make the move on the copy
                initial_hash = self._hash_board(board_copy) # Hash the copied board state

                # Submit the minimax task for this move
                future = executor.submit(self._minimax_memo,
                                         board_copy, # Pass the board copy
                                         self.depth - 1,
                                         False, # Start with minimizing player (opponent)
                                         alpha, beta, # Initial alpha/beta for this branch
                                         initial_hash)
                future_to_move[future] = (r, c)

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_move):
                move = future_to_move[future]
                try:
                    score = future.result()
                    move_results[move] = score
                    print(f"  Move {move} evaluated score: {score} (from thread)")

                    # --- Update Best Move Based on Completed Results ---
                    # Since threads run in parallel, we find the max score from collected results
                    if score > best_score:
                        best_score = score
                        best_move = move
                        print(f"  New best move tentatively: {move} with score {score}")

                    # --- Alpha-Beta Pruning Across Threads (Limited) ---
                    # This is tricky. Simple approach: update alpha based on completed results.
                    # A thread starting later *might* benefit if alpha gets updated significantly
                    # by an earlier finishing thread, but it's not guaranteed like sequential.
                    # alpha = max(alpha, score) # Update alpha globally - careful with races if not managed
                    # More robust parallel alpha-beta is complex (e.g., Young Brothers Wait).
                    # For simplicity here, we primarily rely on pruning within each thread's recursive calls.
                    # The main benefit comes from parallel exploration time saving.

                except Exception as exc:
                    print(f'Move {move} generated an exception: {exc}')
                    move_results[move] = -math.inf # Assign a very low score on error

        # Final determination of best move after all threads are done
        if not best_move and move_results:
             # Find the best move from all results if the incremental update didn't catch it
             # (or if the best score was found by a later thread)
             best_move = max(move_results, key=move_results.get)
             best_score = move_results[best_move]
        elif not best_move and candidate_moves:
            print("Minimax/Threading didn't find a best move, picking first candidate.")
            best_move = candidate_moves[0] # Fallback
        elif not best_move:
             print("Error: No valid moves found after search!")
             return None # Should be handled by fallback in route

        end_time = time.time()
        print(f"AI Calculation time: {end_time - start_time:.2f} seconds")
        print(f"AI chose move: {best_move} with final score: {best_score}")
        return {"y": best_move[0], "x": best_move[1]}


    # Minimax with Alpha-Beta Pruning and Memoization (Thread-safe TT access)
    # Now accepts the board state explicitly
    def _minimax_memo(self, board_state, depth, is_maximizing, alpha, beta, board_hash):
        state_key = (board_hash, depth, is_maximizing)

        # --- Thread-safe Transposition Table Check ---
        with self.tt_lock:
            if state_key in self.transposition_table:
                return self.transposition_table[state_key]
        # --- End Lock ---

        # Base case: depth limit or terminal state
        # Optional: Add explicit check_win checks here for terminal states if needed
        if depth == 0: # Or check for win/loss/draw on board_state
            score = self._evaluate_board(board_state, AI) # Evaluate from AI's perspective
            # --- Thread-safe Transposition Table Store ---
            with self.tt_lock:
                 self.transposition_table[state_key] = score
            # --- End Lock ---
            return score

        current_player = AI if is_maximizing else PLAYER
        moves = self._generate_moves(board_state, current_player)

        if not moves: # No moves left (draw or blocked state)
             with self.tt_lock: self.transposition_table[state_key] = 0 # Store draw score
             return 0

        # Optional: Sort moves within recursion too (consider performance impact)

        best_val = -math.inf if is_maximizing else math.inf

        for r, c in moves:
            board_state[r][c] = current_player # Make move directly on the passed board state
            new_hash = self._update_hash(board_hash, r, c, current_player)

            eval_score = self._minimax_memo(board_state, depth - 1, not is_maximizing, alpha, beta, new_hash)

            board_state[r][c] = EMPTY # Undo move on the passed board state

            if is_maximizing:
                best_val = max(best_val, eval_score)
                alpha = max(alpha, eval_score)
            else:
                best_val = min(best_val, eval_score)
                beta = min(beta, eval_score)

            if beta <= alpha: # Pruning
                break

        # --- Thread-safe Transposition Table Store ---
        with self.tt_lock:
            self.transposition_table[state_key] = best_val
        # --- End Lock ---
        return best_val


    # Heuristic Candidate Move Generation - Now accepts board_state
    def _generate_moves(self, board_state, player_to_check_for):
        # (Keep the _generate_moves logic exactly as it was, but use board_state)
        moves = set()
        has_pieces = False
        radius = 2
        for r in range(SIZE):
            for c in range(SIZE):
                if board_state[r][c] != EMPTY:
                    has_pieces = True
                    for dr in range(-radius, radius + 1):
                        for dc in range(-radius, radius + 1):
                            if dr == 0 and dc == 0: continue
                            nr, nc = r + dr, c + dc
                            if is_valid(board_state, nc, nr) and board_state[nr][nc] == EMPTY:
                                moves.add((nr, nc))
        if not has_pieces:
            center = SIZE // 2
            if board_state[center][center] == EMPTY: return [(center, center)]
            else: return []
        if not moves and has_pieces:
             print("Warning: No neighboring empty cells found in generate_moves.")
        return list(moves)


    # --- Evaluation Function --- Now accepts board_state
    def _evaluate_board(self, board_state, player):
        # (Keep the _evaluate_board logic exactly as it was, but use board_state
        #  and ensure helper functions like _calculate_score_for_player also accept board_state)
        ai_score = self._calculate_score_for_player(board_state, AI)
        player_score = self._calculate_score_for_player(board_state, PLAYER)
        return ai_score - player_score * 1.1 # Prioritize blocking

    def _calculate_score_for_player(self, board_state, player):
        # (Keep the _calculate_score_for_player logic, passing board_state down)
        total_score = 0
        opponent = PLAYER if player == AI else AI
        lines = self._get_all_lines(board_state)
        for line in lines:
            total_score += self._evaluate_line(line, player, opponent) # Assuming _evaluate_line doesn't need board state
        return total_score

    def _get_all_lines(self, board_state):
        # (Keep the _get_all_lines logic, using board_state)
        lines_alt = []
        # Rows
        for r in range(SIZE): lines_alt.append(board_state[r])
        # Columns
        for c in range(SIZE): lines_alt.append([board_state[r][c] for r in range(SIZE)])
        # Diagonals (TL to BR)
        for i in range(-(SIZE - 1), SIZE):
            line = []; r = 0
            for j in range(SIZE): r, c = j, j + i; \
                               if 0 <= r < SIZE and 0 <= c < SIZE: line.append(board_state[r][c])
            if len(line) >= 5: lines_alt.append(line)
        # Anti-diagonals (TR to BL)
        for i in range(2 * SIZE - 1):
            line = []; r = 0
            for j in range(SIZE): r, c = j, i - j; \
                               if 0 <= r < SIZE and 0 <= c < SIZE: line.append(board_state[r][c])
            if len(line) >= 5: lines_alt.append(line)
        return lines_alt

    def _evaluate_line(self, line, player, opponent):
        # (Keep the _evaluate_line logic exactly as it was)
        # ... (rest of the evaluation logic remains the same) ...
        # Simplified placeholder - retain your full pattern matching logic here
        score = 0
        pattern_score = 0
        line_str = "".join(map(str, line))
        p_str = str(player)
        o_str = str(opponent)
        e_str = str(EMPTY)

        # Example patterns (keep your comprehensive list)
        patterns_scores = {
            f"{p_str*5}": SCORE["FIVE"] * 10,
            f"{e_str}{p_str*4}{e_str}": SCORE["LIVE_FOUR"],
            f"{o_str}{p_str*4}{e_str}": SCORE["RUSH_FOUR"],
            f"{e_str}{p_str*4}{o_str}": SCORE["RUSH_FOUR"],
            f"{p_str}{p_str}{e_str}{p_str}{p_str}": SCORE["RUSH_FOUR"] * 0.9,
            f"{p_str}{e_str}{p_str}{p_str}{p_str}": SCORE["RUSH_FOUR"] * 0.9,
            f"{p_str*3}{e_str}{p_str}": SCORE["RUSH_FOUR"] * 0.9,
            f"{e_str}{p_str*3}{e_str}": SCORE["LIVE_THREE"],
            f"{o_str}{p_str*3}{e_str}": SCORE["SLEEP_THREE"],
            f"{e_str}{p_str*3}{o_str}": SCORE["SLEEP_THREE"],
            f"{e_str}{e_str}{p_str*2}{e_str}{e_str}": SCORE["LIVE_TWO"],
            f"{e_str}{p_str}{e_str}{p_str}{e_str}": SCORE["LIVE_TWO"],
            f"{o_str}{p_str*2}{e_str}{e_str}": SCORE["SLEEP_TWO"],
            f"{e_str}{e_str}{p_str*2}{o_str}": SCORE["SLEEP_TWO"],
        }

        for pattern, value in patterns_scores.items():
             # Use finditer for potentially overlapping patterns if needed, or simplify
             if pattern in line_str: # Simple check
                 pattern_score += value

        # Also add simple contiguous score (keep your original logic here too)
        # ... add back the contiguous check loop ...

        return pattern_score # Return the calculated score for the line


# --- Flask Route ---

@app.route('/ai_move', methods=['POST'])
def get_ai_move():
    data = request.json
    if not data or 'board' not in data or 'depth' not in data:
        return jsonify({"error": "Missing board or depth in request"}), 400

    board_state = data['board']
    search_depth = int(data['depth'])

    # Basic validation (keep as is)
    if not isinstance(board_state, list) or len(board_state) != SIZE or \
       not all(isinstance(row, list) and len(row) == SIZE for row in board_state):
        return jsonify({"error": "Invalid board format"}), 400
    if not isinstance(search_depth, int) or search_depth <= 0:
        return jsonify({"error": "Invalid depth"}), 400

    print(f"\nReceived request: Depth={search_depth}")

    # Create AI instance with the current board and depth
    ai = GomokuAI(board_state, search_depth)
    best_move = ai.find_best_move() # This now uses threading

    if best_move:
        return jsonify({"move": best_move})
    else:
        # Fallback if AI completely fails to find a move
        print("Error: AI could not determine a move. Searching for fallback.")
        for r in range(SIZE):
            for c in range(SIZE):
                if board_state[r][c] == EMPTY:
                    print(f"Returning fallback empty spot: ({r},{c})")
                    return jsonify({"move": {"y": r, "x": c}})
        # If still no move, board is full or error state
        return jsonify({"error": "AI failed to find a move (board likely full or internal error)"}), 500

# IMPORTANT: Remove or comment out the following lines for Vercel deployment
# if __name__ == '__main__':
#     # Vercel provides its own server/WSGI interface
#     # app.run(host='0.0.0.0', port=5000, debug=False) # Debug=False for production
#     pass