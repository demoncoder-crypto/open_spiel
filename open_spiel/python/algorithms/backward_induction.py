"""Python implementation of the backward induction algorithm."""

import enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from open_spiel.python import policy as policy_lib
import pyspiel


class TieBreakingPolicy(enum.Enum):
  """Different strategies for breaking ties in backward induction."""

  FIRST_ACTION = 0  # Choose the first action with the best value (default)
  LAST_ACTION = 1  # Choose the last action with the best value
  RANDOM_ACTION = 2  # Choose a random action among the best ones
  ALL_ACTIONS = 3  # Return all actions with the best value


def backward_induction(
    game: pyspiel.Game,
    state: Optional[pyspiel.State] = None,
    tie_breaking_policy: TieBreakingPolicy = TieBreakingPolicy.FIRST_ACTION,
    allow_imperfect_information: bool = False,
) -> Tuple[List[float], Dict[str, int]]:
  """Computes optimal values and policies using backward induction.

  This algorithm computes the optimal values and policy for a perfect
  information game using backward induction. It works for n-player games.

  Args:
    game: The game to analyze.
    state: The state to run from. If None, the initial state is used.
    tie_breaking_policy: How to handle situations where multiple actions have
      the same value. Default is to choose the first action.
    allow_imperfect_information: If True, the algorithm will not validate that
      the game has perfect information, but the result is not guaranteed to be
      a Nash equilibrium.

  Returns:
    A tuple containing:
    - The optimal value for each player at the root
    - The optimal policy (mapping states to actions) for each player
  """
  if game.get_type().dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
    raise ValueError("Backward induction requires sequential games")

  if not allow_imperfect_information:
    if game.get_type().information != pyspiel.GameType.Information.PERFECT_INFORMATION:
      raise ValueError("Backward induction requires perfect information games. "
                    "Use allow_imperfect_information=True to override this check.")
  elif game.get_type().information != pyspiel.GameType.Information.PERFECT_INFORMATION:
    print("WARNING: Running backward induction on an imperfect information "
          "game. The result is NOT guaranteed to be a Nash equilibrium and "
          "should be interpreted carefully.")

  if state is None:
    state = game.new_initial_state()

  # Cache for memoization
  cache = {}

  def get_backward_induction_value(state):
    """Recursive helper function to compute backward induction values."""
    # Use the cache if available
    state_str = str(state)
    if state_str in cache:
      return cache[state_str]

    if state.is_terminal():
      result = (state.returns(), None, [])
      cache[state_str] = result
      return result

    if state.is_chance_node():
      values = np.zeros(game.num_players())
      for action, prob in state.chance_outcomes():
        child = state.child(action)
        child_values, _, _ = get_backward_induction_value(child)
        values += prob * np.array(child_values)
      result = (values.tolist(), None, [])
      cache[state_str] = result
      return result

    # Player node
    current_player = state.current_player()
    legal_actions = state.legal_actions()
    assert legal_actions, f"No legal actions at state: {state}"

    # Start with the first action
    best_action = legal_actions[0]
    child = state.child(best_action)
    best_values, _, _ = get_backward_induction_value(child)
    best_value = best_values[current_player]

    # Track all actions that yield the best value (for tie-breaking)
    best_actions = [best_action]

    # Try all other actions
    for action in legal_actions[1:]:
      child = state.child(action)
      values, _, _ = get_backward_induction_value(child)
      value = values[current_player]

      # If we found a strictly better action
      if value > best_value:
        best_value = value
        best_values = values
        best_action = action
        best_actions = [action]  # Reset best actions list
      # If we found an equally good action
      elif value == best_value:
        best_actions.append(action)
        # Update best_action based on tie-breaking policy
        if tie_breaking_policy == TieBreakingPolicy.LAST_ACTION:
          best_action = action
          best_values = values

    # Handle random tie-breaking if necessary
    if (tie_breaking_policy == TieBreakingPolicy.RANDOM_ACTION and
        len(best_actions) > 1):
      best_action = np.random.choice(best_actions)
      child = state.child(best_action)
      best_values, _, _ = get_backward_induction_value(child)

    result = (best_values, best_action, best_actions)
    cache[state_str] = result
    return result

  # Build the policy
  def build_policy(state, policy_dict):
    """Recursively build the policy dictionary."""
    if state.is_terminal() or state.is_chance_node():
      return

    state_str = str(state)
    _, best_action, _ = cache.get(state_str) or get_backward_induction_value(state)

    if best_action is not None:
      policy_dict[state_str] = best_action

    # Recurse on all children
    for action in state.legal_actions():
      child = state.child(action)
      build_policy(child, policy_dict)

  # Compute values and build policy
  root_values, _, _ = get_backward_induction_value(state)
  policy_dict = {}
  build_policy(state, policy_dict)

  return root_values, policy_dict


def backward_induction_values(game: pyspiel.Game,
                              state: Optional[pyspiel.State] = None) -> List[float]:
  """Returns just the values at the root using backward induction."""
  return backward_induction(game, state)[0]


def backward_induction_all_optimal_actions(
    game: pyspiel.Game,
    state: Optional[pyspiel.State] = None,
    allow_imperfect_information: bool = False,
) -> Tuple[List[float], Dict[str, List[int]]]:
  """Returns all optimal actions when there are ties.

  Args:
    game: The game to analyze.
    state: The state to run from. If None, the initial state is used.
    allow_imperfect_information: If True, the algorithm will not validate that
      the game has perfect information, but the result is not guaranteed to be
      a Nash equilibrium.

  Returns:
    A tuple containing:
    - The optimal value for each player at the root
    - A dictionary mapping states to lists of optimal actions
  """
  if game.get_type().dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
    raise ValueError("Backward induction requires sequential games")

  if not allow_imperfect_information:
    if game.get_type().information != pyspiel.GameType.Information.PERFECT_INFORMATION:
      raise ValueError("Backward induction requires perfect information games. "
                    "Use allow_imperfect_information=True to override this check.")
  elif game.get_type().information != pyspiel.GameType.Information.PERFECT_INFORMATION:
    print("WARNING: Running backward induction on an imperfect information "
          "game. The result is NOT guaranteed to be a Nash equilibrium and "
          "should be interpreted carefully.")

  if state is None:
    state = game.new_initial_state()

  # Cache for memoization
  cache = {}

  def get_backward_induction_value(state):
    """Recursive helper function to compute backward induction values."""
    # Use the cache if available
    state_str = str(state)
    if state_str in cache:
      return cache[state_str]

    if state.is_terminal():
      result = (state.returns(), None, [])
      cache[state_str] = result
      return result

    if state.is_chance_node():
      values = np.zeros(game.num_players())
      for action, prob in state.chance_outcomes():
        child = state.child(action)
        child_values, _, _ = get_backward_induction_value(child)
        values += prob * np.array(child_values)
      result = (values.tolist(), None, [])
      cache[state_str] = result
      return result

    # Player node
    current_player = state.current_player()
    legal_actions = state.legal_actions()
    assert legal_actions, f"No legal actions at state: {state}"

    # Start with the first action
    best_action = legal_actions[0]
    child = state.child(best_action)
    best_values, _, _ = get_backward_induction_value(child)
    best_value = best_values[current_player]

    # Track all actions that yield the best value
    best_actions = [best_action]

    # Try all other actions
    for action in legal_actions[1:]:
      child = state.child(action)
      values, _, _ = get_backward_induction_value(child)
      value = values[current_player]

      # If we found a strictly better action
      if value > best_value:
        best_value = value
        best_values = values
        best_action = action
        best_actions = [action]  # Reset best actions list
      # If we found an equally good action
      elif value == best_value:
        best_actions.append(action)

    result = (best_values, best_action, best_actions)
    cache[state_str] = result
    return result

  # Build the policy with all optimal actions
  def build_policy(state, policy_dict):
    """Recursively build the policy dictionary with all optimal actions."""
    if state.is_terminal() or state.is_chance_node():
      return

    state_str = str(state)
    _, _, best_actions = cache.get(state_str) or get_backward_induction_value(state)

    if best_actions:
      policy_dict[state_str] = best_actions

    # Recurse on all children
    for action in state.legal_actions():
      child = state.child(action)
      build_policy(child, policy_dict)

  # Compute values and build policy
  root_values, _, _ = get_backward_induction_value(state)
  policy_dict = {}
  build_policy(state, policy_dict)

  return root_values, policy_dict


def get_backward_induction_tabular_policy(
    game: pyspiel.Game,
    state: Optional[pyspiel.State] = None,
    tie_breaking_policy: TieBreakingPolicy = TieBreakingPolicy.FIRST_ACTION,
    allow_imperfect_information: bool = False,
) -> policy_lib.TabularPolicy:
  """Returns a TabularPolicy from backward induction.

  Args:
    game: The game to analyze.
    state: The state to run from. If None, the initial state is used.
    tie_breaking_policy: How to handle situations where multiple actions have
      the same value. Default is to choose the first action.
    allow_imperfect_information: If True, the algorithm will not validate that
      the game has perfect information, but the result is not guaranteed to be 
      a Nash equilibrium.

  Returns:
    A TabularPolicy for the game that selects the backward induction
    actions at every state.
  """
  _, policy_dict = backward_induction(
      game, state, tie_breaking_policy, allow_imperfect_information)
  
  # Create an empty tabular policy
  tabular_policy = policy_lib.TabularPolicy(game)
  
  # To convert our policy_dict (which maps state strings to actions) to a TabularPolicy 
  # (which maps information states to action probabilities), we need to traverse 
  # the game tree and collect all relevant states and their optimal actions.
  
  # First, create a mapping from information states to state-action pairs
  info_state_to_actions = {}
  
  # Helper function to traverse the game tree
  def traverse_game_tree(state):
    if state.is_terminal():
      return
      
    if state.is_chance_node():
      # For chance nodes, recurse on all children
      for action, _ in state.chance_outcomes():
        traverse_game_tree(state.child(action))
      return
    
    # For player nodes
    player = state.current_player()
    info_state = state.information_state_string(player)
    legal_actions = state.legal_actions()
    
    # Associate this info state with its legal actions
    if info_state not in info_state_to_actions:
      info_state_to_actions[info_state] = (legal_actions, [])
    
    # Add the state-action pair
    state_str = str(state)
    if state_str in policy_dict:
      best_action = policy_dict[state_str]
      info_state_to_actions[info_state][1].append((state_str, best_action))
    
    # Recurse on all children
    for action in legal_actions:
      traverse_game_tree(state.child(action))
  
  # Start traversal from the given state or initial state
  if state is None:
    state = game.new_initial_state()
  traverse_game_tree(state)
  
  # Now update the TabularPolicy
  for info_state, (legal_actions, state_actions) in info_state_to_actions.items():
    # If we found an optimal action for any state with this info state
    if state_actions:
      action_probs = tabular_policy.policy_for_key(info_state)
      # In case of conflicting optimal actions for different states with 
      # the same info state, we'll use a frequency-based approach
      action_count = {}
      for _, action in state_actions:
        action_count[action] = action_count.get(action, 0) + 1
      
      # Find the most frequently optimal action
      best_action, _ = max(action_count.items(), key=lambda x: x[1])
      
      # Set probability 1.0 for the best action, 0.0 for others
      for action in legal_actions:
        action_probs[action] = 1.0 if action == best_action else 0.0
  
  return tabular_policy 