from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import math
import random
import concurrent.futures # 导入并发库

app = Flask(__name__)
# 允许来自任何源的请求（用于本地开发）
# 生产环境中，应限制为前端的实际来源
# 对于Vercel部署，由于前端和后端在同一域，CORS可能不是必需的，但保留它通常是安全的
CORS(app)

# --- Constants ---
SIZE = 15
PLAYER = 1
AI = 2
EMPTY = 0

# --- Score Constants (与之前相同) ---
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

# --- Game Logic (与之前相同) ---
def is_valid(board, x, y):
    return 0 <= x < SIZE and 0 <= y < SIZE

def check_win(board, player, x, y):
    if not is_valid(board, x, y) or board[y][x] != player:
        return False
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for dx, dy in directions:
        count = 1
        for i in range(1, 5):
            nx, ny = x + i * dx, y + i * dy
            if not is_valid(board, nx, ny) or board[ny][nx] != player: break
            count += 1
        for i in range(1, 5):
            nx, ny = x - i * dx, y - i * dy
            if not is_valid(board, nx, ny) or board[ny][nx] != player: break
            count += 1
        if count >= 5: return True
    return False

# --- AI Core Logic (基本与之前相同) ---
class GomokuAI:
    def __init__(self, board, depth):
        self.board = [row[:] for row in board] # 创建棋盘的深拷贝，防止多线程冲突
        self.depth = depth
        # 注意：如果跨请求共享AI实例或使用全局缓存，需要考虑线程安全
        # 这里每次请求都创建新实例，所以transposition_table是独立的
        self.transposition_table = {}
        # Zobrist表应该定义在类外部或作为类变量，以确保一致性
        # 但为了简单起见，并假设每次请求都独立，暂时保留在这里
        self.zobrist_table = [[[random.randint(1, 2**64 - 1) for _ in range(3)]
                           for _ in range(SIZE)]
                          for _ in range(SIZE)]

    def find_best_move(self):
        # (find_best_move 的内部逻辑与您之前提供的代码相同)
        # ... (包括 _generate_moves, _evaluate_board, _minimax_memo, _hash_board, _update_hash 等)
        # ... (为简洁起见，省略了这部分代码，请使用您上传版本中的完整逻辑)
        start_time = time.time()
        best_score = -math.inf
        best_move = None
        immediate_win_move = None
        immediate_block_move = None

        moves = self._generate_moves(AI)

        # 1. Check for immediate AI win
        for r, c in moves:
            self.board[r][c] = AI
            if check_win(self.board, AI, c, r):
                immediate_win_move = (r, c)
                self.board[r][c] = EMPTY
                break
            self.board[r][c] = EMPTY

        if immediate_win_move:
            print(f"AI found immediate win at {immediate_win_move}")
            return {"y": immediate_win_move[0], "x": immediate_win_move[1]}

        # 2. Check for immediate player win and block
        player_moves = self._generate_moves(PLAYER)
        block_moves = []
        for r, c in player_moves:
            if self.board[r][c] == EMPTY:
                self.board[r][c] = PLAYER
                if check_win(self.board, PLAYER, c, r):
                    block_moves.append((r, c))
                self.board[r][c] = EMPTY

        if len(block_moves) > 0:
            immediate_block_move = block_moves[0]
            print(f"AI blocking player win at {immediate_block_move}")
            if self.board[immediate_block_move[0]][immediate_block_move[1]] == EMPTY:
                 return {"y": immediate_block_move[0], "x": immediate_block_move[1]}
            else:
                 print(f"Warning: Block move {immediate_block_move} target is not empty. Proceeding.")
                 immediate_block_move = None

        # 3. Minimax search
        print(f"Starting Minimax search with depth: {self.depth}")
        alpha = -math.inf
        beta = math.inf

        scored_moves = []
        for r, c in moves:
            self.board[r][c] = AI
            score = self._evaluate_board(AI) # Quick evaluate for sorting
            self.board[r][c] = EMPTY
            scored_moves.append(((r, c), score))
        scored_moves.sort(key=lambda item: item[1], reverse=True)
        sorted_moves = [move for move, score in scored_moves]

        for r, c in sorted_moves:
            self.board[r][c] = AI
            current_hash = self._hash_board()
            score = self._minimax_memo(self.depth - 1, False, alpha, beta, current_hash)
            self.board[r][c] = EMPTY

            print(f"  Move ({r},{c}) evaluated score: {score}")
            if score > best_score:
                best_score = score
                best_move = (r, c)
                print(f"  New best move: ({r},{c}) with score {score}")
            alpha = max(alpha, best_score)
            # Alpha-beta pruning at root level is less common but possible
            # if beta <= alpha:
            #     print(f"Pruning remaining moves at root")
            #     break

        if not best_move and len(moves) > 0:
            print("Minimax didn't find a better move, picking first generated move.")
            best_move = moves[0]
        elif not best_move:
            print("Error: No valid moves found!")
            # Fallback logic to find any empty cell
            for r_idx in range(SIZE):
                for c_idx in range(SIZE):
                    if self.board[r_idx][c_idx] == EMPTY:
                        print(f"Error fallback: picking first empty ({r_idx},{c_idx})")
                        return {"y": r_idx, "x": c_idx}
            return None # Should signal error if board is full or truly no moves

        end_time = time.time()
        print(f"AI Calculation time: {end_time - start_time:.2f} seconds")
        print(f"AI chose move: {best_move} with score: {best_score}")
        if best_move:
             return {"y": best_move[0], "x": best_move[1]}
        else:
             return None # Indicate failure

    # --- Other methods (_minimax_memo, _generate_moves, _evaluate_board, etc.) ---
    # --- Assume these are the same as in your original file                   ---
    # --- Ensure they are correctly placed within the GomokuAI class          ---
    def _minimax_memo(self, depth, is_maximizing, alpha, beta, board_hash):
        state_key = (board_hash, depth, is_maximizing)
        if state_key in self.transposition_table:
            return self.transposition_table[state_key]

        if depth == 0:
            score = self._evaluate_board(AI)
            self.transposition_table[state_key] = score
            return score

        current_player = AI if is_maximizing else PLAYER
        moves = self._generate_moves(current_player)

        if not moves:
             self.transposition_table[state_key] = 0
             return 0

        # Optional move sorting within recursion
        # ...

        if is_maximizing: # AI's turn
            max_eval = -math.inf
            for r, c in moves:
                if self.board[r][c] == EMPTY: # Double check
                    self.board[r][c] = AI
                    new_hash = self._update_hash(board_hash, r, c, AI)
                    eval_score = self._minimax_memo(depth - 1, False, alpha, beta, new_hash)
                    self.board[r][c] = EMPTY
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha: break
            self.transposition_table[state_key] = max_eval
            return max_eval
        else: # Player's turn
            min_eval = math.inf
            for r, c in moves:
                 if self.board[r][c] == EMPTY: # Double check
                    self.board[r][c] = PLAYER
                    new_hash = self._update_hash(board_hash, r, c, PLAYER)
                    eval_score = self._minimax_memo(depth - 1, True, alpha, beta, new_hash)
                    self.board[r][c] = EMPTY
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha: break
            self.transposition_table[state_key] = min_eval
            return min_eval

    def _hash_board(self):
        h = 0
        for r in range(SIZE):
            for c in range(SIZE):
                if self.board[r][c] != EMPTY:
                    player_idx = self.board[r][c] # 1 or 2
                    h ^= self.zobrist_table[r][c][player_idx]
        return h

    def _update_hash(self, current_hash, r, c, player):
         new_hash = current_hash
         # Assumes the spot was empty before the move
         # If overwriting, XOR out the previous piece first
         # new_hash ^= self.zobrist_table[r][c][self.board[r][c]] # If not empty
         new_hash ^= self.zobrist_table[r][c][player] # XOR in the new piece
         return new_hash

    def _generate_moves(self, player_to_check_for):
        moves = set()
        has_pieces = False
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
        if not has_pieces:
            center = SIZE // 2
            return [(center, center)] if self.board[center][center] == EMPTY else []
        return list(moves)

    def _evaluate_board(self, player):
        ai_score = self._calculate_score_for_player(AI)
        player_score = self._calculate_score_for_player(PLAYER)
        return ai_score - player_score * 1.1

    def _calculate_score_for_player(self, player):
        total_score = 0
        opponent = PLAYER if player == AI else AI
        lines = self._get_all_lines()
        for line in lines:
            # Using the more robust pattern matching version
            total_score += self._evaluate_line_patterns(line, player, opponent)
        return total_score

    def _get_all_lines(self):
        lines = []
        # Rows
        for r in range(SIZE): lines.append(self.board[r])
        # Columns
        for c in range(SIZE): lines.append([self.board[r][c] for r in range(SIZE)])
        # Diagonals (TL to BR)
        for i in range(-(SIZE - 5), SIZE - 4): # Only diagonals long enough
            line = []
            for j in range(SIZE):
                r, c = j, j + i
                if 0 <= r < SIZE and 0 <= c < SIZE: line.append(self.board[r][c])
            if len(line) >= 5: lines.append(line)
        # Anti-diagonals (TR to BL)
        for i in range(4, 2 * SIZE - 5): # Only diagonals long enough
            line = []
            for j in range(SIZE):
                r, c = j, i - j
                if 0 <= r < SIZE and 0 <= c < SIZE: line.append(self.board[r][c])
            if len(line) >= 5: lines.append(line)
        return lines

    def _evaluate_line_patterns(self, line, player, opponent):
        # (This is the pattern matching logic from your original code)
        line_str = "".join(map(str, line))
        p_str = str(player)
        o_str = str(opponent)
        e_str = str(EMPTY)

        patterns_scores = {
            f"{p_str*5}": SCORE["FIVE"] * 10,
            f"{e_str}{p_str*4}{e_str}": SCORE["LIVE_FOUR"],
            f"{o_str}{p_str*4}{e_str}": SCORE["RUSH_FOUR"], f"{e_str}{p_str*4}{o_str}": SCORE["RUSH_FOUR"],
            f"{p_str}{p_str}{e_str}{p_str}{p_str}": SCORE["RUSH_FOUR"] * 0.9, f"{p_str}{e_str}{p_str}{p_str}{p_str}": SCORE["RUSH_FOUR"] * 0.9,
            f"{p_str*3}{e_str}{p_str}": SCORE["RUSH_FOUR"] * 0.9, f"{p_str}{e_str}{p_str*3}": SCORE["RUSH_FOUR"] * 0.9, # Added symmetric gapped rush-4
            f"{e_str}{p_str*3}{e_str}": SCORE["LIVE_THREE"],
            f"{o_str}{p_str*3}{e_str}": SCORE["SLEEP_THREE"], f"{e_str}{p_str*3}{o_str}": SCORE["SLEEP_THREE"],
            f"{o_str}{p_str}{e_str}{p_str}{p_str}{e_str}": SCORE["SLEEP_THREE"] * 0.8, f"{e_str}{p_str*2}{e_str}{p_str}{o_str}": SCORE["SLEEP_THREE"] * 0.8, # Corrected second pattern
            f"{o_str}{p_str*2}{e_str}{p_str}{e_str}": SCORE["SLEEP_THREE"] * 0.8, f"{e_str}{p_str}{e_str}{p_str*2}{o_str}": SCORE["SLEEP_THREE"] * 0.8, # Added more sleep-3 gaps
            f"{e_str}{e_str}{p_str*2}{e_str}{e_str}": SCORE["LIVE_TWO"], f"{e_str}{p_str}{e_str}{p_str}{e_str}": SCORE["LIVE_TWO"],
            f"{e_str}{p_str*2}{e_str}{e_str}": SCORE["LIVE_TWO"] * 0.8, f"{e_str}{e_str}{p_str*2}{e_str}": SCORE["LIVE_TWO"] * 0.8,
            f"{o_str}{p_str*2}{e_str}{e_str}": SCORE["SLEEP_TWO"], f"{e_str}{e_str}{p_str*2}{o_str}": SCORE["SLEEP_TWO"],
            f"{o_str}{p_str}{e_str}{p_str}{e_str}": SCORE["SLEEP_TWO"], f"{e_str}{p_str}{e_str}{p_str}{o_str}": SCORE["SLEEP_TWO"],
            f"{e_str}{p_str}{e_str}{p_str}{e_str}": SCORE["SLEEP_TWO"], # O X O X O -> Live Two, Corrected Above
            f"{o_str}{p_str}{e_str}{e_str}{p_str}{e_str}": SCORE["SLEEP_TWO"] * 0.7, # X P _ _ P O
            f"{e_str}{p_str}{e_str}{e_str}{p_str}{o_str}": SCORE["SLEEP_TWO"] * 0.7, # O P _ _ P X
        }

        pattern_score = 0
        # Iterate through possible start positions to find patterns
        # This is more robust than simple `in` check for overlapping patterns
        for pattern, value in patterns_scores.items():
            start_index = 0
            while True:
                index = line_str.find(pattern, start_index)
                if index == -1:
                    break
                pattern_score += value
                start_index = index + 1 # Move past the found pattern start

        # Simple contiguous check (optional, maybe redundant with good patterns)
        # score_contig = self._evaluate_line_contiguous(line, player, opponent)
        # return pattern_score + score_contig * 0.1
        return pattern_score

# --- Create a ThreadPoolExecutor ---
# Limiting to 1 worker means only one AI calculation runs at a time,
# but it runs in a separate thread from the main Flask request handler.
# Increase max_workers if you want concurrent calculations (needs careful resource management).
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# --- Wrapper function to run AI calculation (optional but clean) ---
def run_ai_calculation(board_state, search_depth):
    """Runs the AI calculation in a way suitable for the executor."""
    try:
        ai = GomokuAI(board_state, search_depth)
        return ai.find_best_move()
    except Exception as e:
        print(f"Exception in AI thread: {e}")
        # Propagate exception or return an error indicator
        # Returning None here, the main route will handle it
        return None


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

    print(f"\nReceived request: Depth={search_depth}. Submitting to executor.")

    try:
        # Submit the AI calculation to the thread pool
        # Pass copies of data if needed, though board_state is likely passed by value here anyway
        future = executor.submit(run_ai_calculation, board_state, search_depth)

        # Wait for the result from the future.
        # This still blocks the *current request*, but the main Flask server
        # thread is free to handle other potential incoming requests if needed.
        # Add a timeout if desired: future.result(timeout=60) # e.g., 60 seconds
        best_move = future.result()

        if best_move:
            print(f"AI calculation complete. Move: {best_move}")
            return jsonify({"move": best_move})
        else:
            # This could be due to calculation error or no valid moves found by AI logic
            print("Error: AI calculation failed or returned no move.")
            # Attempt fallback (find any empty spot) - This should be part of run_ai_calculation ideally
            for r in range(SIZE):
                for c in range(SIZE):
                    if board_state[r][c] == EMPTY:
                        print(f"Returning fallback empty spot from main route: ({r},{c})")
                        return jsonify({"move": {"y": r, "x": c}})
            # If still no move, it's likely a draw or error state
            return jsonify({"error": "AI failed to find a move (calculation error or board full?)"}), 500

    except concurrent.futures.TimeoutError:
         print("Error: AI calculation timed out.")
         return jsonify({"error": "AI calculation took too long"}), 504 # Gateway Timeout
    except Exception as e:
        print(f"Error submitting or getting result from AI task: {e}")
        return jsonify({"error": f"Internal server error during AI processing: {e}"}), 500


# Note: The following block is for local execution only.
# Vercel will use the 'app' object directly.
# if __name__ == '__main__':
#     # Use debug=False for production/testing deployment behavior
#     app.run(host='0.0.0.0', port=5000, debug=False)