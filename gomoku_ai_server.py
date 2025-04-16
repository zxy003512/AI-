from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import math
import random  # Used for tie-breaking or fallback
from concurrent.futures import ThreadPoolExecutor, as_completed
import os  # 新增，根据CPU核数设置线程数

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
    "RUSH_FOUR": 100000,  # 包含冲四和活四, 可在一步形成5子时使用
    "LIVE_THREE": 50000,
    "SLEEP_THREE": 1000,
    "LIVE_TWO": 500,
    "SLEEP_TWO": 100,
    "LIVE_ONE": 10,
    "SLEEP_ONE": 1,
    "CENTER_BONUS": 1  # 中心的微小奖励
}

# --- Game Logic Ported to Python ---

def is_valid(board, x, y):
    return 0 <= x < SIZE and 0 <= y < SIZE

def check_win(board, player, x, y):
    if not is_valid(board, x, y) or board[y][x] != player:
        return False

    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # Horizontal, Vertical, Diag \, Diag /
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
        self.transposition_table = {}  # Optional: For memoization

    def find_best_move(self):
        start_time = time.time()
        best_score = -math.inf
        best_move = None
        immediate_win_move = None
        immediate_block_move = None

        moves = self._generate_moves(AI)  # Generate moves for AI (2)

        # 1. Check for immediate AI win
        for r, c in moves:
            self.board[r][c] = AI
            if check_win(self.board, AI, c, r):
                immediate_win_move = (r, c)
                self.board[r][c] = EMPTY  # Undo test move
                break
            self.board[r][c] = EMPTY  # Undo test move

        if immediate_win_move:
            print(f"AI found immediate win at {immediate_win_move}")
            return {"y": immediate_win_move[0], "x": immediate_win_move[1]}

        # 2. Check for immediate player win and block
        player_moves = self._generate_moves(PLAYER)  # Generate potential player moves
        block_moves = []
        for r, c in player_moves:
            if self.board[r][c] == EMPTY:  # Check if cell is empty before testing
                self.board[r][c] = PLAYER
                if check_win(self.board, PLAYER, c, r):
                    block_moves.append((r, c))
                self.board[r][c] = EMPTY  # Undo test move

        if len(block_moves) > 0:
            immediate_block_move = block_moves[0]
            print(f"AI blocking player win at {immediate_block_move}")
            if self.board[immediate_block_move[0]][immediate_block_move[1]] == EMPTY:
                return {"y": immediate_block_move[0], "x": immediate_block_move[1]}
            else:
                print(f"Warning: Block move {immediate_block_move} target is not empty. Proceeding with search.")
                immediate_block_move = None

        # 3. Minimax search if no immediate actions
        print(f"Starting Minimax search with depth: {self.depth}")
        alpha = -math.inf
        beta = math.inf

        # Sort moves based on heuristic evaluation (optional but good for alpha-beta)
        scored_moves = []
        for r, c in moves:
            self.board[r][c] = AI
            score = self._evaluate_board(AI)  # Quick evaluate
            self.board[r][c] = EMPTY
            scored_moves.append(((r, c), score))
        scored_moves.sort(key=lambda item: item[1], reverse=True)
        sorted_moves = [move for move, score in scored_moves]

        # --- Multi-threaded top-level move evaluations ---
        # 根据CPU核心数动态设置线程数
        max_workers = max(1, min(os.cpu_count() or 1, len(sorted_moves)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_move = {executor.submit(self.evaluate_top_move, r, c): (r, c) for r, c in sorted_moves}
            for future in as_completed(future_to_move):
                r, c = future_to_move[future]
                try:
                    score = future.result()
                except Exception as exc:
                    print(f"Move ({r},{c}) generated an exception: {exc}")
                    score = -math.inf
                print(f"  Move ({r},{c}) evaluated score: {score}")
                if score > best_score:
                    best_score = score
                    best_move = (r, c)
                    print(f"  New best move: ({r},{c}) with score {score}")

        if not best_move and len(moves) > 0:
            print("Minimax didn't find a better move, picking first generated move.")
            best_move = moves[0]
        elif not best_move:
            print("Error: No valid moves found!")
            return None

        end_time = time.time()
        print(f"AI Calculation time: {end_time - start_time:.2f} seconds")
        print(f"AI chose move: {best_move} with score: {best_score}")
        return {"y": best_move[0], "x": best_move[1]}

    def evaluate_top_move(self, r, c):
        """
        对于顶层候选走法，深拷贝当前棋盘，在该拷贝上落子后单独构造一个新的AI实例，
        并执行 _minimax_memo 评估最终分数。
        """
        # 深拷贝棋盘（注意：只需要拷贝二维列表即可）
        board_copy = [row[:] for row in self.board]
        board_copy[r][c] = AI
        # 新实例的搜索深度减1
        new_ai = GomokuAI(board_copy, self.depth - 1)
        current_hash = new_ai._hash_board()
        score = new_ai._minimax_memo(new_ai.depth, False, -math.inf, math.inf, current_hash)
        return score

    # Minimax with Alpha-Beta Pruning and Memoization
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

        if is_maximizing:  # AI's turn
            max_eval = -math.inf
            for r, c in moves:
                self.board[r][c] = AI
                new_hash = self._update_hash(board_hash, r, c, AI)
                eval_score = self._minimax_memo(depth - 1, False, alpha, beta, new_hash)
                self.board[r][c] = EMPTY
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cut-off
            self.transposition_table[state_key] = max_eval
            return max_eval
        else:  # Player's turn
            min_eval = math.inf
            for r, c in moves:
                self.board[r][c] = PLAYER
                new_hash = self._update_hash(board_hash, r, c, PLAYER)
                eval_score = self._minimax_memo(depth - 1, True, alpha, beta, new_hash)
                self.board[r][c] = EMPTY
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cut-off
            self.transposition_table[state_key] = min_eval
            return min_eval

    # Transposition Table Hashing (Simple Zobrist-like idea)
    # Initialize random bitstrings for each position and player state
    zobrist_table = [[[random.randint(1, 2**64 - 1) for _ in range(3)]  # 0: Empty, 1: Player, 2: AI
                       for _ in range(SIZE)]
                      for _ in range(SIZE)]

    def _hash_board(self):
        h = 0
        for r in range(SIZE):
            for c in range(SIZE):
                if self.board[r][c] != EMPTY:
                    player_idx = self.board[r][c]  # 1 or 2
                    h ^= self.zobrist_table[r][c][player_idx]
        return h

    def _update_hash(self, current_hash, r, c, player):
        new_hash = current_hash
        new_hash ^= self.zobrist_table[r][c][player]
        return new_hash

    # Heuristic Candidate Move Generation
    def _generate_moves(self, player_to_check_for):
        moves = set()  # Use set to avoid duplicates
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
            if self.board[center][center] == EMPTY:
                return [(center, center)]
            else:
                return []
        if not moves and has_pieces:
            print("Warning: No neighboring empty cells found, generating might be flawed or board state unusual.")
        return list(moves)

    # --- Evaluation Function ---
    def _evaluate_board(self, player):
        ai_score = self._calculate_score_for_player(AI)
        player_score = self._calculate_score_for_player(PLAYER)
        # 略微增加对对方威胁制约的权重（可以认为是进一步增强AI能力的一处调整）
        return ai_score - player_score * 1.1

    def _calculate_score_for_player(self, player):
        total_score = 0
        opponent = PLAYER if player == AI else AI
        lines = self._get_all_lines()
        for line in lines:
            total_score += self._evaluate_line(line, player, opponent)
        return total_score

    def _get_all_lines(self):
        lines_alt = []
        for r in range(SIZE):
            lines_alt.append(self.board[r])
        for c in range(SIZE):
            lines_alt.append([self.board[r][c] for r in range(SIZE)])
        for i in range(-(SIZE - 1), SIZE):
            line = []
            for j in range(SIZE):
                r, c = j, j + i
                if 0 <= r < SIZE and 0 <= c < SIZE:
                    line.append(self.board[r][c])
            if len(line) >= 5: lines_alt.append(line)
        for i in range(2 * SIZE - 1):
            line = []
            for j in range(SIZE):
                r, c = j, i - j
                if 0 <= r < SIZE and 0 <= c < SIZE:
                    line.append(self.board[r][c])
            if len(line) >= 5: lines_alt.append(line)
        return lines_alt

    def _evaluate_line(self, line, player, opponent):
        score = 0
        n = len(line)
        for i in range(n):
            count = 0
            if line[i] == player:
                count = 1
                for j in range(i + 1, n):
                    if line[j] == player:
                        count += 1
                    else:
                        break
                if count >= 5:
                    score += SCORE["FIVE"]
                    continue
                left_empty = (i > 0 and line[i - 1] == EMPTY)
                right_empty = (i + count < n and line[i + count] == EMPTY)
                left_opponent = (i > 0 and line[i - 1] == opponent) or i == 0
                right_opponent = (i + count < n and line[i + count] == opponent) or i + count == n
                if count == 4:
                    if left_empty and right_empty: score += SCORE["LIVE_FOUR"]
                    elif left_empty or right_empty: score += SCORE["RUSH_FOUR"]
                elif count == 3:
                    if left_empty and right_empty:
                        is_blocked_live_three = False
                        if i > 0 and line[i-1] == opponent: is_blocked_live_three=True
                        if i+count < n and line[i+count] == opponent: is_blocked_live_three=True
                        if not is_blocked_live_three:
                            score += SCORE["LIVE_THREE"]
                        else:
                            score += SCORE["SLEEP_THREE"]
                    elif left_empty or right_empty:
                        if left_empty and (i+count < n and line[i+count] == EMPTY): 
                            score += SCORE["SLEEP_THREE"]
                        elif right_empty and (i>0 and line[i-1] == EMPTY): 
                            score += SCORE["SLEEP_THREE"]
                elif count == 2:
                    if left_empty and right_empty:
                        if (i+count+1 < n and line[i+count+1] == EMPTY) or (i > 1 and line[i-2] == EMPTY):
                            score += SCORE["LIVE_TWO"]
                        else:
                            score += SCORE["LIVE_TWO"] / 2
                    elif left_empty or right_empty:
                        if left_empty and (i+count < n and line[i+count] == EMPTY):
                            score += SCORE["SLEEP_TWO"]
                        elif right_empty and (i>0 and line[i-1] == EMPTY):
                            score += SCORE["SLEEP_TWO"]
                        elif left_empty and right_opponent:
                            score += SCORE["SLEEP_TWO"]
                        elif right_empty and left_opponent:
                            score += SCORE["SLEEP_TWO"]
                elif count == 1:
                    if left_empty and right_empty:
                        if (i>1 and line[i-2]==EMPTY) and (i+count+1 < n and line[i+count+1]==EMPTY):
                            score += SCORE["LIVE_ONE"]
                        else:
                            score += SCORE["LIVE_ONE"] / 2
                    elif left_empty or right_empty:
                        score += SCORE["SLEEP_ONE"]
            elif line[i] == EMPTY:
                pass
        line_str = "".join(map(str, line))
        p_str = str(player)
        o_str = str(opponent)
        e_str = str(EMPTY)
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
            f"{o_str}{p_str}{e_str}{p_str}{p_str}{e_str}": SCORE["SLEEP_THREE"] * 0.8,
            f"{e_str}{p_str*2}{e_str}{p_str}{o_str}": SCORE["SLEEP_THREE"] * 0.8,
            f"{e_str}{e_str}{p_str*2}{e_str}{e_str}": SCORE["LIVE_TWO"],
            f"{e_str}{p_str}{e_str}{p_str}{e_str}": SCORE["LIVE_TWO"],
            f"{e_str}{p_str*2}{e_str}{e_str}": SCORE["LIVE_TWO"] * 0.8,
            f"{e_str}{e_str}{p_str*2}{e_str}": SCORE["LIVE_TWO"] * 0.8,
            f"{o_str}{p_str*2}{e_str}{e_str}": SCORE["SLEEP_TWO"],
            f"{e_str}{e_str}{p_str*2}{o_str}": SCORE["SLEEP_TWO"],
            f"{o_str}{p_str}{e_str}{p_str}{e_str}": SCORE["SLEEP_TWO"],
            f"{e_str}{p_str}{e_str}{p_str}{o_str}": SCORE["SLEEP_TWO"],
        }
        pattern_score = 0
        for pattern, value in patterns_scores.items():
            if pattern in line_str:
                pattern_score += value
        return pattern_score + score * 0.1

# --- Flask Route ---

@app.route('/ai_move', methods=['POST'])
def get_ai_move():
    data = request.json
    if not data or 'board' not in data or 'depth' not in data:
        return jsonify({"error": "Missing board or depth in request"}), 400

    board_state = data['board']
    search_depth = int(data['depth'])

    if not isinstance(board_state, list) or len(board_state) != SIZE or \
       not all(isinstance(row, list) and len(row) == SIZE for row in board_state):
        return jsonify({"error": "Invalid board format"}), 400

    if not isinstance(search_depth, int) or search_depth <= 0:
        return jsonify({"error": "Invalid depth"}), 400

    print(f"\nReceived request: Depth={search_depth}")
    ai = GomokuAI(board_state, search_depth)
    best_move = ai.find_best_move()

    if best_move:
        return jsonify({"move": best_move})
    else:
        print("Error: AI could not determine a move.")
        for r in range(SIZE):
            for c in range(SIZE):
                if board_state[r][c] == EMPTY:
                    print(f"Returning fallback empty spot: ({r},{c})")
                    return jsonify({"move": {"y": r, "x": c}})
        return jsonify({"error": "AI failed to find a move (no empty spots or error)"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
