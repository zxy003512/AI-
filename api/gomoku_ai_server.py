from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import math
import random
import concurrent.futures # 引入并发库

app = Flask(__name__)
# 允许所有来源的请求（用于本地开发）
# 生产环境中，应限制为你的前端实际部署的域名
CORS(app)

# --- 常量 ---
SIZE = 15
PLAYER = 1
AI = 2
EMPTY = 0

# 得分常量 (可根据需要调整难度/风格)
SCORE = {
    "FIVE": 10000000,      # 连五
    "LIVE_FOUR": 1000000,   # 活四
    "RUSH_FOUR": 100000,   # 冲四 (包括一步成五的活四和冲四)
    "LIVE_THREE": 50000,    # 活三
    "SLEEP_THREE": 1000,    # 眠三
    "LIVE_TWO": 500,        # 活二
    "SLEEP_TWO": 100,        # 眠二
    "LIVE_ONE": 10,         # 活一
    "SLEEP_ONE": 1,          # 眠一
    "CENTER_BONUS": 1        # 中心的微小加分
}

# --- 游戏逻辑 ---

def is_valid(board, x, y):
    return 0 <= x < SIZE and 0 <= y < SIZE

def check_win(board, player, x, y):
    # (check_win 函数保持不变, 这里省略以节省空间)
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

# --- AI 核心逻辑 (Minimax, 评估) ---

class GomokuAI:
    def __init__(self, board, depth):
        self.board = [row[:] for row in board] # 创建棋盘副本，避免修改原始请求数据
        self.depth = depth
        self.transposition_table = {} # 置换表

    def find_best_move_sync(self): # 重命名，明确是同步执行
        """查找最佳移动的核心逻辑 (同步执行)"""
        start_time = time.time()
        best_score = -math.inf
        best_move = None
        immediate_win_move = None
        immediate_block_move = None

        moves = self._generate_moves(AI) # 为 AI (2) 生成候选步

        # 1. 检查 AI 是否能立即获胜
        for r, c in moves:
            if self.board[r][c] == EMPTY: # 确保是空格
                self.board[r][c] = AI
                if check_win(self.board, AI, c, r):
                    immediate_win_move = (r, c)
                    self.board[r][c] = EMPTY # 撤销测试移动
                    break
                self.board[r][c] = EMPTY # 撤销测试移动

        if immediate_win_move:
            print(f"AI found immediate win at {immediate_win_move}")
            return {"y": immediate_win_move[0], "x": immediate_win_move[1]}

        # 2. 检查玩家是否能立即获胜并进行阻挡
        player_moves = self._generate_moves(PLAYER) # 生成玩家可能的移动
        block_moves = []
        for r, c in player_moves:
            if self.board[r][c] == EMPTY: # 确保是空格
                self.board[r][c] = PLAYER
                if check_win(self.board, PLAYER, c, r):
                    block_moves.append((r, c))
                self.board[r][c] = EMPTY # 撤销测试移动

        if len(block_moves) > 0:
            # 简单处理：阻止第一个发现的威胁点
            immediate_block_move = block_moves[0]
            print(f"AI blocking player win at {immediate_block_move}")
            if self.board[immediate_block_move[0]][immediate_block_move[1]] == EMPTY:
                 return {"y": immediate_block_move[0], "x": immediate_block_move[1]}
            else:
                 print(f"Warning: Block move {immediate_block_move} target is not empty. Proceeding with search.")
                 immediate_block_move = None # 如果目标位置无效则重置

        # 3. 如果没有立即行动，则进行 Minimax 搜索
        print(f"Starting Minimax search with depth: {self.depth}")
        alpha = -math.inf
        beta = math.inf

        # 对候选步进行启发式评估和排序 (可选，但对Alpha-Beta剪枝有利)
        scored_moves = []
        for r, c in moves:
             if self.board[r][c] == EMPTY:
                self.board[r][c] = AI
                score = self._evaluate_board(AI) # 快速评估
                self.board[r][c] = EMPTY
                scored_moves.append(((r, c), score))

        # 按得分降序排序，优先探索高分移动
        scored_moves.sort(key=lambda item: item[1], reverse=True)
        sorted_moves = [move for move, score in scored_moves]

        for r, c in sorted_moves:
            if self.board[r][c] == EMPTY: # 再次确认是空的
                self.board[r][c] = AI
                current_hash = self._hash_board() # 用于置换表
                score = self._minimax_memo(self.depth - 1, False, alpha, beta, current_hash)
                self.board[r][c] = EMPTY # 撤销移动

                print(f"  Move ({r},{c}) evaluated score: {score}") # 调试输出

                if score > best_score:
                    best_score = score
                    best_move = (r, c)
                    print(f"  New best move: ({r},{c}) with score {score}")

                alpha = max(alpha, best_score)
                # 根节点不进行剪枝，探索所有顶级移动
                # if beta <= alpha: break

        if not best_move and len(moves) > 0:
            # 容错处理：如果搜索没有找到最佳移动（可能所有移动得分相同或为负无穷）
            # 则选择第一个有效的候选移动
            valid_moves = [(r, c) for r, c in moves if self.board[r][c] == EMPTY]
            if valid_moves:
                print("Minimax didn't find a decisively better move, picking first valid generated move.")
                best_move = valid_moves[0]
            else:
                 print("Error: No valid moves left in generated list!")
                 best_move = None # 表示没有有效移动
        elif not best_move:
             print("Error: No valid moves found at all!")
             best_move = None

        end_time = time.time()
        print(f"AI Calculation time: {end_time - start_time:.2f} seconds")

        if best_move:
            print(f"AI chose move: {best_move} with score: {best_score}")
            return {"y": best_move[0], "x": best_move[1]}
        else:
            # 极端情况：真的没有地方可下（棋盘满或逻辑错误）
             print("AI failed to find any valid move.")
             return None # 返回 None 表示失败

    # Minimax (带 Alpha-Beta 剪枝和置换表)
    def _minimax_memo(self, depth, is_maximizing, alpha, beta, board_hash):
        # ( _minimax_memo 函数保持不变, 这里省略以节省空间)
        state_key = (board_hash, depth, is_maximizing)
        if state_key in self.transposition_table:
            return self.transposition_table[state_key]

        if depth == 0:
            score = self._evaluate_board(AI) # Evaluate from AI's perspective
            self.transposition_table[state_key] = score
            return score

        current_player = AI if is_maximizing else PLAYER
        moves = self._generate_moves(current_player)
        valid_moves = [(r, c) for r, c in moves if self.board[r][c] == EMPTY] #确保只处理空位

        if not valid_moves: # No valid moves left
             self.transposition_table[state_key] = self._evaluate_board(AI) # Return current board score if no moves
             return self.transposition_table[state_key]


        if is_maximizing: # AI's turn
            max_eval = -math.inf
            for r, c in valid_moves: # Use sorted_moves if sorting enabled
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
            for r, c in valid_moves: # Use sorted_moves if sorting enabled
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

    # --- 置换表哈希 (Zobrist Hashing) ---
    # 初始化随机位串 (放在类外部或作为类变量)
    zobrist_table = [[[random.randint(1, 2**64 - 1) for _ in range(3)] # 0: Empty, 1: Player, 2: AI
                      for _ in range(SIZE)]
                     for _ in range(SIZE)]

    def _hash_board(self):
        # ( _hash_board 函数保持不变, 省略)
        h = 0
        for r in range(SIZE):
            for c in range(SIZE):
                if self.board[r][c] != EMPTY:
                    player_idx = self.board[r][c] # 1 or 2
                    h ^= self.zobrist_table[r][c][player_idx]
        return h

    def _update_hash(self, current_hash, r, c, player):
        # ( _update_hash 函数保持不变, 省略)
        new_hash = current_hash
        new_hash ^= self.zobrist_table[r][c][player]
        return new_hash

    # --- 启发式候选步生成 ---
    def _generate_moves(self, player_to_check_for):
        # ( _generate_moves 函数基本不变, 确保只返回空位, 省略)
        moves = set()
        has_pieces = False
        radius = 2 # 考虑现有棋子周围2格范围内的空位

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

        # 如果棋盘为空，下在中心
        if not has_pieces:
            center = SIZE // 2
            if self.board[center][center] == EMPTY:
               return [(center, center)]
            else: # 如果中心已被占（理论上在空棋盘时不应发生）
                # 作为备选，返回棋盘上的第一个空位
                for r in range(SIZE):
                    for c in range(SIZE):
                         if self.board[r][c] == EMPTY:
                             return [(r, c)]
                return [] # 如果完全没有空位

        # 如果周围没有空位（不太可能，除非棋盘快满了），可以考虑所有空位（效率低）
        if not moves and has_pieces:
             print("Warning: No neighboring empty cells found.")
             # Fallback: Add all empty cells
             all_empty = set()
             for r in range(SIZE):
                 for c in range(SIZE):
                     if self.board[r][c] == EMPTY:
                         all_empty.add((r, c))
             return list(all_empty)


        return list(moves)


    # --- 评估函数 ---
    def _evaluate_board(self, player):
        # ( _evaluate_board 函数保持不变, 省略)
        ai_score = self._calculate_score_for_player(AI)
        player_score = self._calculate_score_for_player(PLAYER)
        # 稍微增加对手得分的权重，以倾向于防守
        return ai_score - player_score * 1.1

    def _calculate_score_for_player(self, player):
        # ( _calculate_score_for_player 函数保持不变, 省略)
        total_score = 0
        opponent = PLAYER if player == AI else AI
        lines = self._get_all_lines()
        for line in lines:
            total_score += self._evaluate_line(line, player, opponent)
        return total_score

    def _get_all_lines(self):
        # ( _get_all_lines 函数使用较简洁的版本, 保持不变, 省略)
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
        return lines_alt


    def _evaluate_line(self, line, player, opponent):
        # ( _evaluate_line 函数使用基于模式匹配的版本, 保持不变, 省略)
        # ... (使用基于 patterns_scores 的模式匹配逻辑) ...
        score = 0 # 可以保留基础分或完全依赖模式分
        line_str = "".join(map(str, line))
        p_str = str(player)
        o_str = str(opponent)
        e_str = str(EMPTY)

        patterns_scores = {
            # 五子连珠
            f"{p_str*5}": SCORE["FIVE"] * 10,
            # 活四: OXXXXO
            f"{e_str}{p_str*4}{e_str}": SCORE["LIVE_FOUR"],
            # 冲四: XOOOO_, _OOOOX, OO_OO, O_OOO etc.
            f"{o_str}{p_str*4}{e_str}": SCORE["RUSH_FOUR"],
            f"{e_str}{p_str*4}{o_str}": SCORE["RUSH_FOUR"],
            f"{p_str}{p_str}{e_str}{p_str}{p_str}": SCORE["RUSH_FOUR"] * 0.9,
            f"{p_str}{e_str}{p_str}{p_str}{p_str}": SCORE["RUSH_FOUR"] * 0.9,
            f"{p_str*3}{e_str}{p_str}": SCORE["RUSH_FOUR"] * 0.9,
            # 活三: OXXXO
            f"{e_str}{p_str*3}{e_str}": SCORE["LIVE_THREE"],
            # 眠三: XOOO_, _OOOX, XO_OO, XOO_O etc.
            f"{o_str}{p_str*3}{e_str}": SCORE["SLEEP_THREE"],
            f"{e_str}{p_str*3}{o_str}": SCORE["SLEEP_THREE"],
            f"{o_str}{p_str}{e_str}{p_str}{p_str}{e_str}": SCORE["SLEEP_THREE"] * 0.8,
            f"{e_str}{p_str*2}{e_str}{p_str}{o_str}": SCORE["SLEEP_THREE"] * 0.8,
             # 特殊眠三 O_XXX_O or O_X_XX_O
            f"{e_str}{p_str}{e_str}{p_str*2}{e_str}": SCORE["SLEEP_THREE"],
            f"{e_str}{p_str*2}{e_str}{p_str}{e_str}": SCORE["SLEEP_THREE"],
            # 活二: OXXO O_X_XO OOXXO OXXOO
            f"{e_str}{e_str}{p_str*2}{e_str}{e_str}": SCORE["LIVE_TWO"],
            f"{e_str}{p_str}{e_str}{p_str}{e_str}": SCORE["LIVE_TWO"],
            f"{e_str}{p_str*2}{e_str}{e_str}": SCORE["LIVE_TWO"] * 0.8,
            f"{e_str}{e_str}{p_str*2}{e_str}": SCORE["LIVE_TWO"] * 0.8,
            # 眠二: XOO__, __OOX, XO_O_, _O_OX etc.
            f"{o_str}{p_str*2}{e_str}{e_str}": SCORE["SLEEP_TWO"],
            f"{e_str}{e_str}{p_str*2}{o_str}": SCORE["SLEEP_TWO"],
            f"{o_str}{p_str}{e_str}{p_str}{e_str}": SCORE["SLEEP_TWO"],
            f"{e_str}{p_str}{e_str}{p_str}{o_str}": SCORE["SLEEP_TWO"],
             f"{o_str}{e_str}{p_str*2}{e_str}": SCORE["SLEEP_TWO"] * 0.5, # X O XX O
             f"{e_str}{p_str*2}{e_str}{o_str}": SCORE["SLEEP_TWO"] * 0.5, # O XX O X
        }

        pattern_score = 0
        # 更精确的计数，避免重叠模式的重复计分 (示例)
        counted_indices = set()
        # 优先匹配长模式
        sorted_patterns = sorted(patterns_scores.keys(), key=len, reverse=True)

        for pattern in sorted_patterns:
            value = patterns_scores[pattern]
            start_index = 0
            while True:
                idx = line_str.find(pattern, start_index)
                if idx == -1:
                    break
                # 检查此匹配是否与已计数的索引重叠
                overlaps = False
                for i in range(idx, idx + len(pattern)):
                    if i in counted_indices:
                        overlaps = True
                        break
                if not overlaps:
                    pattern_score += value
                    for i in range(idx, idx + len(pattern)):
                        counted_indices.add(i)
                    start_index = idx + len(pattern) # 从匹配结束后继续查找
                else:
                    start_index = idx + 1 # 从下一个位置开始查找

        return pattern_score # 返回基于模式匹配的得分


# 创建一个线程池执行器
# max_workers 可以根据服务器CPU核心数调整，None通常表示使用核心数*5
executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)

# --- Flask 路由 ---

@app.route('/ai_move', methods=['POST'])
def get_ai_move():
    data = request.json
    if not data or 'board' not in data or 'depth' not in data:
        return jsonify({"error": "Missing board or depth in request"}), 400

    board_state = data['board']
    search_depth = int(data['depth'])

    # 基本验证
    if not isinstance(board_state, list) or len(board_state) != SIZE or \
       not all(isinstance(row, list) and len(row) == SIZE for row in board_state):
        return jsonify({"error": "Invalid board format"}), 400
    if not isinstance(search_depth, int) or search_depth <= 0:
        return jsonify({"error": "Invalid depth"}), 400

    print(f"\nReceived request: Depth={search_depth}. Submitting to background thread.")

    # 创建 AI 实例
    # 注意：board_state 需要传递给 AI 实例，确保线程安全（AI类内部已做拷贝）
    ai = GomokuAI(board_state, search_depth)

    # 提交 AI 计算任务到线程池
    # lambda: ai.find_best_move_sync() 确保在后台线程中调用正确的函数
    future = executor.submit(lambda: ai.find_best_move_sync())

    try:
        # 等待后台线程计算结果
        # future.result() 会阻塞当前请求处理线程，直到计算完成
        # 但它允许 Flask 服务器（如果配置为多工作进程/线程模式）处理其他并发请求
        best_move = future.result()

        if best_move:
            print(f"Background task completed. Returning move: {best_move}")
            return jsonify({"move": best_move})
        else:
            # AI 无法找到移动 (可能是棋盘满了或者内部错误)
            print("Error: AI task completed but returned no valid move.")
            # 尝试返回第一个可用的空位作为最后的补救措施
            for r in range(SIZE):
                for c in range(SIZE):
                    if board_state[r][c] == EMPTY:
                        print(f"Returning fallback empty spot: ({r},{c})")
                        return jsonify({"move": {"y": r, "x": c}})
            # 如果连空位都找不到
            return jsonify({"error": "AI failed to find a move (board full or internal error)"}), 500

    except Exception as e:
        # 捕获后台任务中可能发生的任何异常
        print(f"Error during AI calculation in background thread: {e}")
        # 可以记录更详细的错误信息，例如使用 logging 模块
        # import traceback
        # traceback.print_exc()
        return jsonify({"error": f"AI calculation failed: {e}"}), 500

# 注意：下面的 if __name__ == '__main__': ... 块仅用于本地开发
# Vercel 或其他生产 WSGI 服务器（如 Gunicorn）会直接导入 'app' 对象，不会执行这里面的 app.run()
# 因此，不需要在此修改端口或 host
if __name__ == '__main__':
    # 运行开发服务器
    # host='0.0.0.0' 允许局域网访问, port=5000 是默认端口
    # debug=False 在生产环境中更安全稳定
    app.run(host='0.0.0.0', port=5000, debug=False)

# Vercel 需要能够导入这个 'app' 变量
# Python 文件名是 gomoku_ai_server.py，Vercel配置中会用到