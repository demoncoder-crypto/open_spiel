#include "open_spiel/algorithms/backward_induction.h"

#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <random>
#include <iostream>

namespace open_spiel {
namespace algorithms {
namespace {

// Use OpenSpiel's random number generation instead of std::mt19937
// The random tie-breaking policy will now use this

// Memoized version of the backward induction algorithm
BackwardInductionResult ComputeBackwardInduction(
    const State& state, 
    std::unordered_map<std::string, BackwardInductionResult>& cache,
    TieBreakingPolicy tie_breaking_policy,
    std::shared_ptr<rng_t> rng) {
  
  // Check if this state has already been computed
  std::string state_str = state.ToString();
  auto it = cache.find(state_str);
  if (it != cache.end()) {
    return it->second;
  }

  // Handle terminal states
  if (state.IsTerminal()) {
    // Get the returns for all players
    auto returns = state.Returns();
    // Validate returns
    SPIEL_CHECK_EQ(returns.size(), state.NumPlayers());
    BackwardInductionResult result{returns, kInvalidAction, {}};
    cache[state_str] = result;
    return result;
  }

  // Handle chance nodes
  if (state.IsChanceNode()) {
    // For chance nodes, compute the expected value
    std::vector<double> values(state.NumPlayers(), 0.0);
    // Get chance outcomes
    auto chance_outcomes = state.ChanceOutcomes();
    // Validate there are chance outcomes
    SPIEL_CHECK_GT(chance_outcomes.size(), 0);
    for (const auto& [action, prob] : chance_outcomes) {
      // Validate probability is valid
      SPIEL_CHECK_GE(prob, 0.0);
      SPIEL_CHECK_LE(prob, 1.0);
      
      std::unique_ptr<State> child = state.Child(action);
      // Recursively compute expected values for child states
      auto child_values = ComputeBackwardInduction(*child, cache, tie_breaking_policy, rng).values;
      // Accumulate expected values
      for (Player p = 0; p < values.size(); ++p) {
        values[p] += prob * child_values[p];
      }
    }
    BackwardInductionResult result{values, kInvalidAction, {}};
    cache[state_str] = result;
    return result;
  }

  // For player nodes, compute the value that maximizes that player's payoff
  Player current_player = state.CurrentPlayer();
  // Validate current player is valid
  SPIEL_CHECK_GE(current_player, 0);
  SPIEL_CHECK_LT(current_player, state.NumPlayers());
  
  std::vector<Action> legal_actions = state.LegalActions();
  SPIEL_CHECK_GT(legal_actions.size(), 0);

  // Start with the first action as best
  Action best_action = legal_actions[0];
  std::unique_ptr<State> first_child = state.Child(best_action);
  std::vector<double> best_values = ComputeBackwardInduction(*first_child, cache, tie_breaking_policy, rng).values;
  // Validate best_values
  SPIEL_CHECK_EQ(best_values.size(), state.NumPlayers());
  double best_value = best_values[current_player];
  
  // Track all actions that yield the best value (for tie-breaking)
  std::vector<Action> best_actions{best_action};

  // Try all other actions
  for (int i = 1; i < legal_actions.size(); ++i) {
    Action action = legal_actions[i];
    std::unique_ptr<State> child = state.Child(action);
    auto result = ComputeBackwardInduction(*child, cache, tie_breaking_policy, rng);
    double value = result.values[current_player];

    // If we found a strictly better action
    if (value > best_value) {
      best_value = value;
      best_values = result.values;
      best_action = action;
      best_actions.clear();  // Clear previous best actions
      best_actions.push_back(action);
    } 
    // If we found an equally good action
    else if (value == best_value) {
      best_actions.push_back(action);
      
      // Update best_action based on tie-breaking policy
      if (tie_breaking_policy == TieBreakingPolicy::kLastAction) {
        best_action = action;
        best_values = result.values;
      }
    }
  }
  
  // Handle random tie-breaking if necessary
  if (tie_breaking_policy == TieBreakingPolicy::kRandomAction && best_actions.size() > 1) {
    // Use OpenSpiel's RNG instead of std::uniform_int_distribution
    int idx = rng->RandomInt(0, best_actions.size() - 1);
    best_action = best_actions[idx];
    
    // Need to recompute the values for the randomly chosen action
    std::unique_ptr<State> child = state.Child(best_action);
    auto result = ComputeBackwardInduction(*child, cache, tie_breaking_policy, rng);
    best_values = result.values;
  }

  BackwardInductionResult result{best_values, best_action, best_actions};
  cache[state_str] = result;
  return result;
}

}  // namespace

std::pair<std::vector<double>, std::map<std::string, Action>> BackwardInduction(
    const Game& game, const State* state, TieBreakingPolicy tie_breaking_policy,
    bool allow_imperfect_information) {
  
  // Verify this is a sequential game
  GameType game_type = game.GetType();
  SPIEL_CHECK_EQ(game_type.dynamics, GameType::Dynamics::kSequential);
  
  // Verify this is a perfect information game or explicitly allowed imperfect information
  if (game_type.information != GameType::Information::kPerfectInformation && 
      !allow_imperfect_information) {
    SpielFatalError(
        "Backward induction requires a perfect information game. "
        "Use allow_imperfect_information=true to override, but note "
        "that the result may not be a Nash equilibrium.");
  }
  
  // Print warning if running on imperfect information game
  if (game_type.information != GameType::Information::kPerfectInformation && 
      allow_imperfect_information) {
    std::cerr << "WARNING: Running backward induction on an imperfect "
              << "information game. The result is NOT guaranteed to be "
              << "a Nash equilibrium and should be interpreted carefully." 
              << std::endl;
  }
  
  // Verify game is finite (required for backward induction)
  SPIEL_CHECK_TRUE(game.MaxGameLength() < INT_MAX);
  
  // Create root state if needed
  std::unique_ptr<State> root;
  if (state == nullptr) {
    root = game.NewInitialState();
    state = root.get();
  } else {
    // Verify state belongs to this game
    SPIEL_CHECK_TRUE(state != nullptr);
    SPIEL_CHECK_EQ(state->GetGame()->ToString(), game.ToString());
  }

  // Cache for memoization
  std::unordered_map<std::string, BackwardInductionResult> cache;
  
  // Create an RNG with a fixed seed for reproducibility
  auto rng = std::make_shared<rng_t>(/*seed=*/42);

  // Run backward induction and build policy
  std::map<std::string, Action> policy;
  std::function<void(const State&)> build_policy = [&](const State& s) {
    if (s.IsTerminal() || s.IsChanceNode()) return;

    // Use the cache if available, otherwise compute
    std::string s_str = s.ToString();
    BackwardInductionResult result;
    auto it = cache.find(s_str);
    if (it != cache.end()) {
      result = it->second;
    } else {
      result = ComputeBackwardInduction(s, cache, tie_breaking_policy, rng);
    }

    if (result.best_action != kInvalidAction) {
      policy[s_str] = result.best_action;
    }

    // Recurse on all children to build complete policy
    for (Action action : s.LegalActions()) {
      std::unique_ptr<State> child = s.Child(action);
      build_policy(*child);
    }
  };

  build_policy(*state);
  auto root_result = ComputeBackwardInduction(*state, cache, tie_breaking_policy, rng);
  return {root_result.values, policy};
}

std::vector<double> BackwardInductionValues(const Game& game,
                                          const State* state) {
  return BackwardInduction(game, state).first;
}

std::pair<std::vector<double>, std::map<std::string, std::vector<Action>>>
BackwardInductionAllOptimalActions(const Game& game, const State* state, 
                                  bool allow_imperfect_information) {
  // Verify this is a sequential game
  GameType game_type = game.GetType();
  SPIEL_CHECK_EQ(game_type.dynamics, GameType::Dynamics::kSequential);
  
  // Verify this is a perfect information game or explicitly allowed imperfect information
  if (game_type.information != GameType::Information::kPerfectInformation && 
      !allow_imperfect_information) {
    SpielFatalError(
        "Backward induction requires a perfect information game. "
        "Use allow_imperfect_information=true to override, but note "
        "that the result may not be a Nash equilibrium.");
  }
  
  // Print warning if running on imperfect information game
  if (game_type.information != GameType::Information::kPerfectInformation && 
      allow_imperfect_information) {
    std::cerr << "WARNING: Running backward induction on an imperfect "
              << "information game. The result is NOT guaranteed to be "
              << "a Nash equilibrium and should be interpreted carefully." 
              << std::endl;
  }
  
  // Verify game is finite (required for backward induction)
  SPIEL_CHECK_TRUE(game.MaxGameLength() < INT_MAX);
  
  // Create root state if needed
  std::unique_ptr<State> root;
  if (state == nullptr) {
    root = game.NewInitialState();
    state = root.get();
  } else {
    // Verify state belongs to this game
    SPIEL_CHECK_TRUE(state != nullptr);
    SPIEL_CHECK_EQ(state->GetGame()->ToString(), game.ToString());
  }

  // Cache for memoization
  std::unordered_map<std::string, BackwardInductionResult> cache;
  TieBreakingPolicy tie_breaking_policy = TieBreakingPolicy::kAllActions;
  
  // Create an RNG with a fixed seed for reproducibility
  auto rng = std::make_shared<rng_t>(/*seed=*/42);

  // Run backward induction and build policy of all optimal actions
  std::map<std::string, std::vector<Action>> policy;
  std::function<void(const State&)> build_policy = [&](const State& s) {
    if (s.IsTerminal() || s.IsChanceNode()) return;

    // Use the cache if available, otherwise compute
    std::string s_str = s.ToString();
    BackwardInductionResult result;
    auto it = cache.find(s_str);
    if (it != cache.end()) {
      result = it->second;
    } else {
      result = ComputeBackwardInduction(s, cache, tie_breaking_policy, rng);
    }

    if (!result.best_actions.empty()) {
      policy[s_str] = result.best_actions;
    }

    // Recurse on all children to build complete policy
    for (Action action : s.LegalActions()) {
      std::unique_ptr<State> child = s.Child(action);
      build_policy(*child);
    }
  };

  build_policy(*state);
  auto root_result = ComputeBackwardInduction(*state, cache, tie_breaking_policy, rng);
  return {root_result.values, policy};
}

}  // namespace algorithms
}  // namespace open_spiel 