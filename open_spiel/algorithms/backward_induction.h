#ifndef OPEN_SPIEL_ALGORITHMS_BACKWARD_INDUCTION_H_
#define OPEN_SPIEL_ALGORITHMS_BACKWARD_INDUCTION_H_

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"

namespace open_spiel {
namespace algorithms {

// Enum for different tie-breaking strategies
enum class TieBreakingPolicy {
  kFirstAction,    // Choose the first action with the best value (default)
  kLastAction,     // Choose the last action with the best value
  kRandomAction,   // Choose a random action among the best ones
  kAllActions      // Return all actions with the best value
};

// A struct to hold the result of backward induction:
// - value: the game value for each player at a state
// - best_action: the optimal action for the current player at a state
// - best_actions: all optimal actions if TieBreakingPolicy::kAllActions is used
struct BackwardInductionResult {
  std::vector<double> values;  // One value per player
  Action best_action;
  std::vector<Action> best_actions;  // Only filled when using kAllActions policy
};

// Computes optimal values and policies for a perfect information game using
// backward induction. Works for n-player games.
// Returns a pair containing:
// - The optimal value for each player at the root
// - The optimal policy (mapping states to actions) for each player
// The tie_breaking_policy parameter determines how to handle situations
// where multiple actions have the same value.
// If allow_imperfect_information is true, the algorithm will not validate
// that the game has perfect information, but the result is not guaranteed
// to be a Nash equilibrium.
std::pair<std::vector<double>, std::map<std::string, Action>> BackwardInduction(
    const Game& game,
    const State* state = nullptr,
    TieBreakingPolicy tie_breaking_policy = TieBreakingPolicy::kFirstAction,
    bool allow_imperfect_information = false);

// Helper function that returns just the values at the root
std::vector<double> BackwardInductionValues(
    const Game& game,
    const State* state = nullptr);

// Version that returns all optimal actions when there are ties
// If allow_imperfect_information is true, the algorithm will not validate
// that the game has perfect information, but the result is not guaranteed
// to be a Nash equilibrium.
std::pair<std::vector<double>, std::map<std::string, std::vector<Action>>>
BackwardInductionAllOptimalActions(
    const Game& game, 
    const State* state = nullptr, 
    bool allow_imperfect_information = false);

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_BACKWARD_INDUCTION_H_ 