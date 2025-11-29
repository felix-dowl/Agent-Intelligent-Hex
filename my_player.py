from math import inf
import heapq
import time

from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_hex import GameStateHex
from seahorse.utils.custom_exceptions import MethodNotImplementedError

BRIDGE_PATTERNS = [ # tuple -> (target coord), [coords of the two intermediate cells]
    ((-2, 1), [(-1, 0), (-1, 1)]), # up above : up left and up right 
    ((-1, -1), [(-1, 0), (0, -1)]), # upper left: up left and left
    ((-1, 2), [(-1, 1), (0, 1)]), # upper right: up right and right
    ((1, -2), [(0, -1), (1, -1)]), # lower left: down left and left
    ((1, 1), [(0, 1), (1, 0)]), # lower right: down right and right
    ((2, -1), [(1, 0), (1, -1)]), # down: down right + down left
]

INITIAL_NODES_EXPLORED = 20
BASE_CLIPPING_DEPTH = 2
EARLY_CLIPPING_DEPTH = 1
MAX_DEPTH = 5
MIDGAME_DEPTH = 3
LOW_TIME_DEPTH = 3
EARLY_DEPTH = 2
THIRTY_SEC = 30
SIXTY_SEC = 60
THREE_MIN = 180

class MyPlayer(PlayerHex):
    """
    Player class for Hex game

    Attributes:
        piece_type (str): piece type of the player "R" for the first player and "B" for the second player
    """

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        """
        Initialize the PlayerHex instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
        """
        super().__init__(piece_type, name)

    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Picks a action using minimax with alpha-beta pruning, with depth limiter right now.
        Has a first move heuristic to take a center piece on the first move.
        """
        start_time = time.perf_counter()
        env = current_state.get_rep().get_env()
        actions = tuple(current_state.get_possible_light_actions())
        step = current_state.get_step() if hasattr(current_state, "get_step") else len(env)
        dim = current_state.get_rep().get_dimensions()[0]
        depth = MIDGAME_DEPTH
        clipping_depth = BASE_CLIPPING_DEPTH
        my_piece_count = sum(1 for piece in env.values() if piece.get_type() == self.piece_type)


        #Opening strategy, play in center
        center_candidates = [
            (dim//2 - 1, dim//2 - 1), (dim//2 - 1, dim//2),
            (dim//2, dim//2 - 1), (dim//2, dim//2)
        ]
        if my_piece_count == 0:
            for act in actions:
                pos = act.data.get("position")
                if pos in center_candidates and pos not in env:
                    return act

        # Panic mode: skip search when very low on time and pick the best path-heuristic move.
        if remaining_time <= THIRTY_SEC:
            return self._panic_action(current_state, actions)


        # Early-game depth limiting: after our first move, next three moves search shallow
        if 1 <= my_piece_count <= 3:
            depth = EARLY_DEPTH
            clipping_depth = EARLY_CLIPPING_DEPTH

        # Time-aware depth adjustments.
        if remaining_time <= SIXTY_SEC:
            depth = EARLY_DEPTH
            clipping_depth = EARLY_CLIPPING_DEPTH
        elif remaining_time <= THREE_MIN:
            depth = LOW_TIME_DEPTH

        #Run minimax implementation
        score, action = self.maxAction(current_state, depth, clipping_depth, -inf, inf, start_time, remaining_time)
        if self._time_remaining(start_time, remaining_time) <= THIRTY_SEC:
            return self._panic_action(current_state, actions)
        if action is None:
            possible_actions = tuple(current_state.get_possible_light_actions())
            if not possible_actions:
                raise MethodNotImplementedError("No possible actions to play.")
            return possible_actions[0]
        return action

    def maxAction(self, current_state: GameState, depth: int, clipping_depth: int, alpha: float, beta: float, start_time: float, total_time: float) -> tuple[float, Action | None]:
        if self._time_remaining(start_time, total_time) <= THIRTY_SEC:
            return inf, None
        if depth <= 0 or current_state.is_done():
            return self.evaluate_state(current_state), None

        actions = tuple(current_state.get_possible_light_actions())
        if not actions:
            return self.evaluate_state(current_state), None

        ordered_actions = self._order_actions(current_state, actions, depth, clipping_depth)

        best_value = -inf
        best_action = None
        for action in ordered_actions:
            child_state = current_state.apply_action(action)
            value, _ = self.minAction(child_state, depth - 1, clipping_depth, alpha, beta, start_time, total_time)
            if value > best_value:
                best_value = value
                best_action = action
                alpha = max(alpha, best_value)
                if best_value >= beta:
                    break
        return best_value, best_action

    def minAction(self, current_state: GameState, depth: int, clipping_depth: int, alpha: float, beta: float, start_time: float, total_time: float) -> tuple[float, Action | None]:
        if self._time_remaining(start_time, total_time) <= THIRTY_SEC:
            return -inf, None
        if depth <= 0 or current_state.is_done():
            return self.evaluate_state(current_state), None

        actions = tuple(current_state.get_possible_light_actions())
        if not actions:
            return self.evaluate_state(current_state), None

        ordered_actions = self._order_actions(current_state, actions, depth, clipping_depth)

        best_value = inf
        best_action = None
        for action in ordered_actions:
            child_state = current_state.apply_action(action)
            value, _ = self.maxAction(child_state, depth - 1, clipping_depth, alpha, beta, start_time, total_time)
            if value < best_value:
                best_value = value
                best_action = action
                beta = min(beta, best_value)
                if best_value <= alpha:
                    break
        return best_value, best_action

    def evaluate_state(self, state: GameState) -> float:
        my_id, opp_id = self._find_player_ids(state)
        if state.is_done():
            return 1e6 * (state.scores.get(my_id, 0.0) - state.scores.get(opp_id, 0.0))

        my_cost = self._shortest_path_cost(state, my_id)
        opp_cost = self._shortest_path_cost(state, opp_id) if opp_id is not None else inf

        if my_cost == inf and opp_cost == inf:
            return 0.0
        # lower path cost is better; flip to a score where higher is better for us
        return opp_cost - my_cost
    
    def _shortest_path_cost(self, state: GameStateHex, pid: int) -> float:
        """
        Djikstra shortest-path algorithm between the two sides.
        Own stones cost 0, empty cells cost 1, opponent stones cost infinity.
        Also usues a pattern search to find a certain unblockable bridge pattern. -> oXo shape
        To add : possibly a way to prioritise the middle control in early game?
        """
        player = state.get_player_id(pid)
        if player is None:
            return inf
        piece_type = player.get_piece_type()

        dim = state.get_rep().get_dimensions()[0]
        env = state.get_rep().get_env()
        block_cost = inf

        # precompute enemy bridge cells (empty cells that complete an opponent bridge).
        # need to be computed because they are impassible
        enemy_type = "R" if piece_type == "B" else "B"
        enemy_bridge_cells: set[tuple[int, int]] = set()
        for (i, j), piece in env.items():
            if piece.get_type() != enemy_type:
                continue
            for (dr, dc), mids in BRIDGE_PATTERNS:
                target = (i + dr, j + dc)
                if not state.in_board(target):
                    continue
                target_piece = env.get(target)
                if target_piece is None or target_piece.get_type() != enemy_type:
                    continue
                mid_positions = [(i + mr, j + mc) for mr, mc in mids]
                if any(not state.in_board(mid) for mid in mid_positions):
                    continue
                if all(env.get(mid) is None for mid in mid_positions):
                    enemy_bridge_cells.update(mid_positions)

        # Thois is to determine the cost of the cell during djikstra
        def cell_cost(pos):
            piece = env.get(pos)
            if piece is None:
                return 3.0 if pos in enemy_bridge_cells else 1.0
            if piece.get_type() == piece_type:
                return 0.0
            return block_cost #infinity (ennemy piece)

        # Orientation: players[0] tires top & bottom, players[1] left & right
        vertical = state.players and pid == state.players[0].id
        if vertical:
            starts = [(0, j) for j in range(dim)]
            targets = {(dim - 1, j) for j in range(dim)}
        else:
            starts = [(i, 0) for i in range(dim)]
            targets = {(i, dim - 1) for i in range(dim)}

        dist: dict[tuple[int, int], float] = {}
        heap: list[tuple[float, tuple[int, int]]] = []
        for pos in starts:
            cost = cell_cost(pos)
            if cost >= block_cost:
                continue
            dist[pos] = cost
            heapq.heappush(heap, (cost, pos))

        while heap:
            d, pos = heapq.heappop(heap)
            if d != dist.get(pos, inf):
                continue
            if pos in targets:
                return d
            for _, (ni, nj) in state.get_neighbours(*pos).values():
                npos = (ni, nj)
                step = cell_cost(npos)
                if step >= block_cost:
                    continue
                nd = d + step
                if nd < dist.get(npos, inf):
                    dist[npos] = nd
                    heapq.heappush(heap, (nd, npos))

            # Bridges: oXo pattern where it is technically unblockable
            # Set to a low weight of 0.5 cause its not fully connected yet but better than empty cells
            # ref: left: (0, -1), right (0, 1), up-left: (-1, 0), up-right: (-1, 1), down-left: (1, -1), down-right: (1, 0)
            piece_here = env.get(pos)
            if piece_here is not None and piece_here.get_type() == piece_type:

                for (dr, dc), mids in BRIDGE_PATTERNS:
                    tpos = (pos[0] + dr, pos[1] + dc)
                    if not state.in_board(tpos):
                        continue
                    target_piece = env.get(tpos)
                    if target_piece is None or target_piece.get_type() != piece_type:
                        continue
                    if any(env.get((pos[0] + mr, pos[1] + mc)) is not None for mr, mc in mids):
                        continue
                    nd = d + 0.5 
                    if nd < dist.get(tpos, inf):
                        dist[tpos] = nd
                        heapq.heappush(heap, (nd, tpos))
        return inf

    def _bridge_defenses(self, state, my_type) -> list[tuple[int,int]]:
        """
        Returns empty cell of any bridge that is being attacked
        """
        env = state.get_rep().get_env()
        defenses = []
        for (i,j), piece in env.items():
            if piece.get_type() != my_type: continue
            for (dr,dc), mids in BRIDGE_PATTERNS:
                tpos = (i+dr, j+dc)
                tp = env.get(tpos)
                if tp is None or tp.get_type() != my_type: continue
                mid1, mid2 = [(i+mr, j+mc) for mr,mc in mids]
                m1, m2 = env.get(mid1), env.get(mid2)
                if m1 is None and m2 is None:
                    continue
                if m1 is None and m2 is not None and m2.get_type() != my_type:
                    defenses.append(mid1)
                if m2 is None and m1 is not None and m1.get_type() != my_type:
                    defenses.append(mid2)
        return defenses

    def _panic_action(self, state: GameState, actions: tuple[Action, ...]) -> Action:
        """
        Fast fallback: choose the move that maximizes our path advantage without deeper search.
        """
        if not actions:
            raise MethodNotImplementedError("No possible actions to play.")

        my_id, opp_id = self._find_player_ids(state)
        best_score = -inf
        best_action = actions[0]
        for action in actions:
            child_state = state.apply_action(action)
            my_cost = self._shortest_path_cost(child_state, my_id)
            opp_cost = self._shortest_path_cost(child_state, opp_id) if opp_id is not None else inf
            if my_cost == inf and opp_cost == inf:
                score = 0.0
            else:
                score = opp_cost - my_cost
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _order_actions(self, current_state: GameState, actions: tuple[Action, ...], depth: int, clipping_depth: int) -> list[Action]:
        """
        Orders actions by a heuristic score and injects bridge defenses into the shortlist.
        """
        if depth < clipping_depth:
            return list(actions)

        my_id, opp_id = self._find_player_ids(current_state)
        my_before = self._shortest_path_cost(current_state, my_id)
        opp_before = self._shortest_path_cost(current_state, opp_id) if opp_id is not None else inf

        scored: list[tuple[float, Action]] = []
        for action in actions:
            child_state = current_state.apply_action(action)
            my_after = self._shortest_path_cost(child_state, my_id)
            opp_after = self._shortest_path_cost(child_state, opp_id) if opp_id is not None else inf
            delta = 0.0
            if my_before < inf:
                delta += my_before - my_after
            if opp_before < inf and opp_after < inf:
                delta += opp_after - opp_before
            scored.append((delta, action))

        scored.sort(key=lambda x: x[0], reverse=True)
        k = min(INITIAL_NODES_EXPLORED, len(scored))
        top_actions = [a for _, a in scored[:k]]

        # Make sure bridge defenses are explored even if they fall outside the top-k.
        move_piece = actions[0].data.get("piece") if actions else None
        defense_positions = self._bridge_defenses(current_state, move_piece) if move_piece else []
        if not defense_positions:
            return top_actions

        action_by_pos = {a.data["position"]: a for a in actions}
        defense_actions = [action_by_pos[pos] for pos in defense_positions if pos in action_by_pos]

        ordered: list[Action] = []
        seen_positions: set[tuple[int, int]] = set()
        for act in defense_actions + top_actions:
            pos = act.data["position"]
            if pos in seen_positions:
                continue
            ordered.append(act)
            seen_positions.add(pos)

        return ordered

    def _time_remaining(self, start_time: float, total_time: float) -> float:
        return total_time - (time.perf_counter() - start_time)

    # returns the tuple [my_id, opp_id]
    def _find_player_ids(self, state: GameState) -> tuple[int, int]:
        my_id = None
        opp_id = None
        for player in getattr(state, "players", []):
            if player.get_piece_type() == self.piece_type:
                my_id = player.get_id()
            else:
                opp_id = player.get_id()
        if my_id == None:
            my_id = self.get_id()
        if opp_id is None and hasattr(state, "scores"):
            for pid in state.scores:
                if pid != my_id:
                    opp_id = pid
                    break
        return my_id, opp_id
