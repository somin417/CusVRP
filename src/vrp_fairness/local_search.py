"""
Fairness improvement algorithm using local search.
"""

import random
import copy
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .metrics import calculate_solution_metrics, SolutionMetrics


@dataclass
class Move:
    """Represents a local search move."""
    move_type: str  # "relocate" or "swap"
    route_idx: int
    stop_idx: int
    target_route_idx: Optional[int] = None
    target_stop_idx: Optional[int] = None
    score_delta: float = 0.0


class FairnessLocalSearch:
    """Local search algorithm for fairness improvement."""
    
    def __init__(
        self,
        routes: List[Dict[str, Any]],
        stops_dict: Dict[str, Dict[str, Any]],
        time_matrix: Dict[tuple, int],
        baseline_cost: float,
        eps: float = 0.10,
        lambda_balance: float = 0.1,
        depot_id: str = "Depot"
    ):
        """
        Initialize local search.
        
        Args:
            routes: List of route dictionaries
            stops_dict: Dictionary mapping stop_id to stop data
            time_matrix: Dictionary mapping (stop1_id, stop2_id) to travel time
            baseline_cost: Baseline total cost
            eps: Cost budget tolerance (1+eps)
            lambda_balance: Balance penalty weight
            depot_id: ID of the depot
        """
        self.routes = copy.deepcopy(routes)
        self.stops_dict = stops_dict
        self.time_matrix = time_matrix
        self.baseline_cost = baseline_cost
        self.cost_budget = baseline_cost * (1 + eps)
        self.lambda_balance = lambda_balance
        self.depot_id = depot_id
        self.iterations = 0
    
    def _calculate_score(self, routes: List[Dict[str, Any]]) -> Tuple[float, SolutionMetrics]:
        """
        Calculate objective score for a solution.
        
        Returns:
            (score, metrics) where score = W_max + lambda_balance * imbalance
        """
        metrics = calculate_solution_metrics(routes, self.stops_dict, self.time_matrix, self.depot_id)
        
        # Imbalance: max - min route duration
        imbalance = metrics.driver_balance
        
        # Score: minimize max waiting time + balance penalty
        score = metrics.W_max + self.lambda_balance * imbalance
        
        return score, metrics
    
    def _get_worst_wait_stop(self, routes: List[Dict[str, Any]]) -> Optional[Tuple[int, int, str]]:
        """
        Find the stop with worst waiting time.
        
        Returns:
            (route_idx, stop_idx, stop_id) or None
        """
        worst_wait = -1.0
        worst_stop = None
        
        for route_idx, route in enumerate(routes):
            stop_ids = route["ordered_stop_ids"]
            current_time = 0.0
            
            for stop_idx in range(1, len(stop_ids) - 1):  # Skip depot at start/end
                stop_id = stop_ids[stop_idx]
                
                # Calculate arrival time
                from_stop = stop_ids[stop_idx - 1]
                travel_time = self.time_matrix.get((from_stop, stop_id), 0)
                current_time += travel_time
                
                if current_time > worst_wait:
                    worst_wait = current_time
                    worst_stop = (route_idx, stop_idx, stop_id)
                
                # Add service time
                service_time = self.stops_dict.get(stop_id, {}).get("service_time", 300)
                current_time += service_time
        
        return worst_stop
    
    def _apply_relocate(
        self,
        routes: List[Dict[str, Any]],
        route_idx: int,
        stop_idx: int,
        target_route_idx: int,
        target_pos: int
    ) -> List[Dict[str, Any]]:
        """Apply relocate move."""
        new_routes = copy.deepcopy(routes)
        
        # Remove stop from source route
        stop_id = new_routes[route_idx]["ordered_stop_ids"].pop(stop_idx)
        
        # Insert into target route
        new_routes[target_route_idx]["ordered_stop_ids"].insert(target_pos, stop_id)
        
        return new_routes
    
    def _apply_swap(
        self,
        routes: List[Dict[str, Any]],
        route_idx1: int,
        stop_idx1: int,
        route_idx2: int,
        stop_idx2: int
    ) -> List[Dict[str, Any]]:
        """Apply swap move."""
        new_routes = copy.deepcopy(routes)
        
        stop_id1 = new_routes[route_idx1]["ordered_stop_ids"][stop_idx1]
        stop_id2 = new_routes[route_idx2]["ordered_stop_ids"][stop_idx2]
        
        new_routes[route_idx1]["ordered_stop_ids"][stop_idx1] = stop_id2
        new_routes[route_idx2]["ordered_stop_ids"][stop_idx2] = stop_id1
        
        return new_routes
    
    def _generate_candidate_moves(
        self,
        route_idx: int,
        stop_idx: int,
        k: int = 10
    ) -> List[Move]:
        """Generate K candidate moves for a stop."""
        moves = []
        stop_ids = self.routes[route_idx]["ordered_stop_ids"]
        stop_id = stop_ids[stop_idx]
        
        # Skip if it's the depot
        if stop_id == self.depot_id:
            return moves
        
        # Generate relocate moves
        for target_route_idx in range(len(self.routes)):
            target_route = self.routes[target_route_idx]
            target_stop_ids = target_route["ordered_stop_ids"]
            
            # Try inserting at different positions
            for target_pos in range(1, len(target_stop_ids)):  # Skip first depot
                if target_route_idx == route_idx and (target_pos == stop_idx or target_pos == stop_idx + 1):
                    continue  # Skip invalid positions
                
                moves.append(Move(
                    move_type="relocate",
                    route_idx=route_idx,
                    stop_idx=stop_idx,
                    target_route_idx=target_route_idx,
                    target_stop_idx=target_pos
                ))
        
        # Generate swap moves
        for target_route_idx in range(len(self.routes)):
            target_route = self.routes[target_route_idx]
            target_stop_ids = target_route["ordered_stop_ids"]
            
            for target_stop_idx in range(1, len(target_stop_ids) - 1):  # Skip depots
                if target_route_idx == route_idx and target_stop_idx == stop_idx:
                    continue  # Same stop
                
                moves.append(Move(
                    move_type="swap",
                    route_idx=route_idx,
                    stop_idx=stop_idx,
                    target_route_idx=target_route_idx,
                    target_stop_idx=target_stop_idx
                ))
        
        # Sample K moves if too many
        if len(moves) > k:
            moves = random.sample(moves, k)
        
        return moves
    
    def _evaluate_move(self, move: Move) -> Tuple[float, float, List[Dict[str, Any]]]:
        """
        Evaluate a move.
        
        Returns:
            (score, cost, new_routes)
        """
        if move.move_type == "relocate":
            new_routes = self._apply_relocate(
                self.routes,
                move.route_idx,
                move.stop_idx,
                move.target_route_idx,
                move.target_stop_idx
            )
        elif move.move_type == "swap":
            new_routes = self._apply_swap(
                self.routes,
                move.route_idx,
                move.stop_idx,
                move.target_route_idx,
                move.target_stop_idx
            )
        else:
            return float('inf'), float('inf'), self.routes
        
        # Calculate new score and cost
        score, metrics = self._calculate_score(new_routes)
        cost = metrics.total_cost
        
        return score, cost, new_routes
    
    def improve(self, max_iters: int = 300) -> Tuple[List[Dict[str, Any]], SolutionMetrics, int]:
        """
        Run local search improvement.
        
        Returns:
            (improved_routes, final_metrics, iterations)
        """
        current_score, current_metrics = self._calculate_score(self.routes)
        best_routes = copy.deepcopy(self.routes)
        best_score = current_score
        
        for iteration in range(max_iters):
            self.iterations = iteration + 1
            
            # Find worst-wait stop
            worst_stop = self._get_worst_wait_stop(self.routes)
            if worst_stop is None:
                break
            
            route_idx, stop_idx, stop_id = worst_stop
            
            # Generate candidate moves
            moves = self._generate_candidate_moves(route_idx, stop_idx, k=20)
            
            if not moves:
                break
            
            # Evaluate moves and find best improving move
            best_move = None
            best_move_score = current_score
            best_move_cost = current_metrics.total_cost
            best_move_routes = None
            
            for move in moves:
                score, cost, new_routes = self._evaluate_move(move)
                
                # Check cost budget
                if cost > self.cost_budget:
                    continue
                
                # Check if improving
                if score < best_move_score:
                    best_move = move
                    best_move_score = score
                    best_move_cost = cost
                    best_move_routes = new_routes
            
            # Apply best move if found
            if best_move is not None and best_move_routes is not None:
                self.routes = best_move_routes
                current_score = best_move_score
                current_metrics = calculate_solution_metrics(
                    self.routes, self.stops_dict, self.time_matrix, self.depot_id
                )
                
                if current_score < best_score:
                    best_score = current_score
                    best_routes = copy.deepcopy(self.routes)
            else:
                # No improving move found
                break
        
        final_metrics = calculate_solution_metrics(
            best_routes, self.stops_dict, self.time_matrix, self.depot_id
        )
        
        return best_routes, final_metrics, self.iterations

