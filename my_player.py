from math import inf
import heapq

from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_hex import GameStateHex
from seahorse.utils.custom_exceptions import MethodNotImplementedError

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
        env = current_state.get_rep().get_env()
        actions = tuple(current_state.get_possible_light_actions())
        step = current_state.get_step() if hasattr(current_state, "get_step") else len(env)
        dim = current_state.get_rep().get_dimensions()[0]
        center_candidates = [
            (dim//2 - 1, dim//2 - 1), (dim//2 - 1, dim//2),
            (dim//2, dim//2 - 1), (dim//2, dim//2)
        ]
        # opening heuristics, play in center
        if step == 0 or step == 1:
            for act in actions:
                pos = act.data.get("position")
                if pos in center_candidates and pos not in env:
                    return act
        depth = 2  # to be changed for dynamic with time ad with goodness of move
        score, action = self.maxAction(current_state, depth, -inf, inf)
        if action is None:
            possible_actions = tuple(current_state.get_possible_light_actions())
            if not possible_actions:
                raise MethodNotImplementedError("No possible actions to play.")
            return possible_actions[0]
        return action

    def maxAction(self, current_state: GameState, depth: int, alpha: float, beta: float) -> tuple[float, Action | None]:
        if depth <= 0 or current_state.is_done():
            return self.evaluate_state(current_state), None

        actions = tuple(current_state.get_possible_light_actions())
        if not actions:
            return self.evaluate_state(current_state), None

        best_value = -inf
        best_action = None
        for action in actions:
            child_state = current_state.apply_action(action)
            value, _ = self.minAction(child_state, depth - 1, alpha, beta)
            if value > best_value:
                best_value = value
                best_action = action
                alpha = max(alpha, best_value)
                if best_value >= beta:
                    break
        return best_value, best_action

    def minAction(self, current_state: GameState, depth: int, alpha: float, beta: float) -> tuple[float, Action | None]:
        if depth <= 0 or current_state.is_done():
            return self.evaluate_state(current_state), None

        actions = tuple(current_state.get_possible_light_actions())
        if not actions:
            return self.evaluate_state(current_state), None

        best_value = inf
        best_action = None
        for action in actions:
            child_state = current_state.apply_action(action)
            value, _ = self.maxAction(child_state, depth - 1, alpha, beta)
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

        # Thois is to determine the cost of the cell during djikstra
        def cell_cost(pos):
            piece = env.get(pos)
            if piece is None:
                return 1.0
            if piece.get_type() == piece_type:
                return 0.0
            return block_cost #infinity (ennemy piece)

        # Orientation: players[0] tires top & bottom, players[1] left & right inchaallah
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
                bridge_patterns = [ # tuple -> (target coord), [coords of the two intermediate cells]
                    ((-2, 1), [(-1, 0), (-1, 1)]), # up above : up left and up right 
                    ((-1, -1), [(-1, 0), (0, -1)]), # upper left: up left and left
                    ((-1, 2), [(-1, 1), (0, 1)]), # upper right: up right and right
                    ((1, -2), [(0, -1), (1, -1)]), # lower left: down left and left
                    ((1, 1), [(0, 1), (1, 0)]), # lower right: down right and right
                    ((2, -1), [(1, 0), (1, -1)]), # down: down right + down left
                ]
                for (dr, dc), mids in bridge_patterns:
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
