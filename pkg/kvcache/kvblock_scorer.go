/*
Copyright 2025 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package kvcache

import (
	"context"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
)

// KVScoringStrategy defines the strategy used to score pods for KV cache block reuse.
type KVScoringStrategy string

const (
	// LongestPrefixMatch Score by longest consecutive match from start.
	LongestPrefixMatch KVScoringStrategy = "LongestPrefix"
	// HybridPrefixMatch Score for HMA models with separate full attention and SWA scoring.
	HybridPrefixMatch KVScoringStrategy = "HybridPrefix"
)

// KVBlockScorerConfig holds the configuration for the KVBlockScorer.
type KVBlockScorerConfig struct {
	ScoringStrategy KVScoringStrategy
	BackendConfigs  []*KVCacheBackendConfig `json:"backendConfigs"`
	ModelRegistry   *ModelRegistry          `json:"-"`
}

// DefaultKVBlockScorerConfig returns the default configuration for the KVBlockScorer.
func DefaultKVBlockScorerConfig() *KVBlockScorerConfig {
	return &KVBlockScorerConfig{
		ScoringStrategy: LongestPrefixMatch,
		BackendConfigs:  DefaultKVCacheBackendConfig(),
	}
}

// KVBlockScorer defines the interface for implementing a KV block scoring
// strategy.
type KVBlockScorer interface {
	// Strategy returns the scoring strategy type.
	Strategy() KVScoringStrategy
	// Score scores the blocks based on the scoring strategy.
	// modelName is used by HMA scorers to determine attention group configuration.
	// It returns a map of pod names to their scores.
	Score(ctx context.Context, keys []kvblock.BlockHash,
		keyToPods map[kvblock.BlockHash][]kvblock.PodEntry, modelName string) (map[string]float64, error)
}

// NewKVBlockScorer creates a new KVBlockScorer based on the provided strategy.
func NewKVBlockScorer(config *KVBlockScorerConfig) (KVBlockScorer, error) {
	switch config.ScoringStrategy {
	case LongestPrefixMatch:
		// Build weight map from list of BackendConfigs for efficient lookup
		weightMap := make(map[string]float64)
		for _, medium := range config.BackendConfigs {
			weightMap[medium.Name] = medium.Weight
		}

		return &LongestPrefixScorer{
			MediumWeights: weightMap,
		}, nil
	case HybridPrefixMatch:
		// Build weight map from list of BackendConfigs for efficient lookup
		weightMap := make(map[string]float64)
		for _, medium := range config.BackendConfigs {
			weightMap[medium.Name] = medium.Weight
		}

		if config.ModelRegistry == nil {
			return nil, fmt.Errorf("model registry required for HybridPrefixMatch strategy")
		}

		return &HybridPrefixCacheScorer{
			MediumWeights: weightMap,
			ModelRegistry: config.ModelRegistry,
		}, nil
	default:
		return nil, fmt.Errorf("unsupported scoring strategy: %s", config.ScoringStrategy)
	}
}

// LongestPrefixScorer scores based on longest consecutive block matches count
// starting from block 0.
type LongestPrefixScorer struct {
	// mediumWeights maps medium/device tier names to their scoring weights
	MediumWeights map[string]float64
}

// Strategy returns the strategy type: LongestPrefixMatch.
func (s *LongestPrefixScorer) Strategy() KVScoringStrategy {
	return LongestPrefixMatch
}

// fillMaxWeights populates dst with the maximum weight per podID across all
// device tiers for the given entries. The caller must clear dst before calling.
func fillMaxWeights(dst map[string]float64, entries []kvblock.PodEntry, mediumWeights map[string]float64) {
	for _, entry := range entries {
		weight := 1.0
		if mediumWeights != nil {
			if w, exists := mediumWeights[entry.DeviceTier]; exists {
				weight = w
			}
		}
		if cur, exists := dst[entry.PodIdentifier]; !exists || weight > cur {
			dst[entry.PodIdentifier] = weight
		}
	}
}

// Score implements the longest prefix scoring logic with weighted sum based on BackendConfig.
func (s *LongestPrefixScorer) Score(
	_ context.Context,
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
	_ string, // modelName not used for simple prefix scoring
) (map[string]float64, error) {
	if len(keys) == 0 {
		return make(map[string]float64), nil
	}

	podScores := make(map[string]float64)

	// Scratch map reused across iterations to avoid per-key allocation.
	curWeights := make(map[string]float64)

	// Build weight index for the first key in a single pass over entries.
	fillMaxWeights(curWeights, keyToPods[keys[0]], s.MediumWeights)

	// activePods tracks pods still in the consecutive prefix chain.
	// Using a plain map and in-place deletion avoids allocating new sets
	// on every iteration.
	activePods := make(map[string]struct{}, len(curWeights))
	for pod, w := range curWeights {
		activePods[pod] = struct{}{}
		podScores[pod] = w
	}

	for i := 1; i < len(keys); i++ {
		if len(activePods) == 0 {
			break
		}

		// Reuse scratch map: clear and refill for current key.
		clear(curWeights)
		fillMaxWeights(curWeights, keyToPods[keys[i]], s.MediumWeights)

		// In-place intersection: delete pods from activePods that are not
		// in the current key, and accumulate scores for those that remain.
		for pod := range activePods {
			if w, exists := curWeights[pod]; exists {
				podScores[pod] += w
			} else {
				delete(activePods, pod)
			}
		}
	}

	// Return the map containing the final score for each pod encountered.
	return podScores, nil
}

// HybridPrefixCacheScorer scores HMA models with multiple attention groups.
// Each group is scored independently based on its attention type:
// - Full attention: prefix matching (left-to-right from start), multiplied by MULTIPLIER
// - Sliding window: suffix matching (right-to-left from end, limited by window size)
// Scoring uses magnitude separation: fullScore × MULTIPLIER + swaScore
// This ensures full attention always dominates, with sliding window as tiebreaker.
type HybridPrefixCacheScorer struct {
	MediumWeights map[string]float64
	ModelRegistry *ModelRegistry
}

const (
	// fullAttentionMultiplier ensures full attention score dominates sliding window score.
	// Must be > (max_window_size × max_weight). Using 100,000 for safety.
	fullAttentionMultiplier = 100000.0
)

// Strategy returns the strategy type: HybridPrefixMatch.
func (s *HybridPrefixCacheScorer) Strategy() KVScoringStrategy {
	return HybridPrefixMatch
}

// Score implements hybrid scoring for HMA models.
func (s *HybridPrefixCacheScorer) Score(
	ctx context.Context,
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
	modelName string,
) (map[string]float64, error) {
	if len(keys) == 0 {
		return make(map[string]float64), nil
	}

	// Check if model uses HMA
	// If not in registry or IsHMA=false → use simple prefix scoring
	isHMA := s.ModelRegistry.IsHMA(modelName)
	if !isHMA {
		// TODO(tmp): Debug logging - REMOVE after debugging
		logger := log.FromContext(ctx)
		logger.Info("[TMP-DEBUG] HybridScorer: Model not HMA, using simple prefix",
			"modelName", modelName)
		return s.scoreSimplePrefix(ctx, keys, keyToPods)
	}

	// Get attention groups for HMA model
	attentionGroups := s.ModelRegistry.GetAttentionGroups(modelName)
	if len(attentionGroups) == 0 {
		// Model is HMA but no groups configured - log warning and fallback
		// TODO(tmp): Debug logging - REMOVE after debugging
		logger := log.FromContext(ctx)
		logger.Info("[TMP-DEBUG] HybridScorer: HMA model but no groups configured, fallback",
			"modelName", modelName)
		return s.scoreSimplePrefix(ctx, keys, keyToPods)
	}

	// TODO(tmp): Debug logging - REMOVE after debugging
	logger := log.FromContext(ctx)
	logger.Info("[TMP-DEBUG] HybridScorer: Scoring HMA model",
		"modelName", modelName,
		"numGroups", len(attentionGroups),
		"numBlocks", len(keys))

	podScores := make(map[string]float64)

	// Score each attention group independently
	// Apply multiplier immediately when saving to avoid extra loop
	for _, group := range attentionGroups {
		switch group.AttentionType {
		case AttentionTypeFull:
			// Full attention: score from start (prefix matching)
			groupScore := s.scoreFullAttentionGroup(keys, keyToPods, group.GroupID)
			// Apply multiplier immediately: ensures full attention dominates
			for pod, score := range groupScore {
				podScores[pod] += score * fullAttentionMultiplier
			}
			// TODO(tmp): Debug logging - REMOVE after debugging
			logger.Info("[TMP-DEBUG] HybridScorer: Full attention group scored",
				"groupID", group.GroupID,
				"numPods", len(groupScore))
		case AttentionTypeSlidingWindow:
			// Sliding window: score from end (suffix matching within window)
			groupScore := s.scoreSlidingWindowGroup(keys, keyToPods, group.GroupID, group.SlidingWindowSize)
			// Add SWA score directly (acts as tiebreaker)
			for pod, score := range groupScore {
				podScores[pod] += score
			}
			// TODO(tmp): Debug logging - REMOVE after debugging
			logger.Info("[TMP-DEBUG] HybridScorer: Sliding window group scored",
				"groupID", group.GroupID,
				"windowSize", group.SlidingWindowSize,
				"numPods", len(groupScore))
		default:
			continue
		}
	}

	return podScores, nil
}

// scoreSimplePrefix is a fallback for models without attention group configuration.
func (s *HybridPrefixCacheScorer) scoreSimplePrefix(
	_ context.Context,
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
) (map[string]float64, error) {
	podScores := make(map[string]float64)
	curWeights := make(map[string]float64)

	fillMaxWeights(curWeights, keyToPods[keys[0]], s.MediumWeights)

	activePods := make(map[string]struct{}, len(curWeights))
	for pod, w := range curWeights {
		activePods[pod] = struct{}{}
		podScores[pod] = w
	}

	for i := 1; i < len(keys); i++ {
		if len(activePods) == 0 {
			break
		}

		clear(curWeights)
		fillMaxWeights(curWeights, keyToPods[keys[i]], s.MediumWeights)

		for pod := range activePods {
			if w, exists := curWeights[pod]; exists {
				podScores[pod] += w
			} else {
				delete(activePods, pod)
			}
		}
	}

	return podScores, nil
}

// scoreFullAttentionGroup scores full attention group with prefix matching (left-to-right).
func (s *HybridPrefixCacheScorer) scoreFullAttentionGroup(
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
	groupID int,
) map[string]float64 {
	podScores := make(map[string]float64)
	curWeights := make(map[string]float64)

	// Filter entries to only those containing this group
	firstEntries := filterByGroup(keyToPods[keys[0]], groupID)
	// TODO(tmp): Debug logging - REMOVE after debugging
	fmt.Printf("[TMP-DEBUG] scoreFullAttentionGroup: blockIdx=0, groupID=%d, totalEntries=%d, filteredEntries=%d\n",
		groupID, len(keyToPods[keys[0]]), len(firstEntries))
	fillMaxWeights(curWeights, firstEntries, s.MediumWeights)

	activePods := make(map[string]struct{}, len(curWeights))
	for pod, w := range curWeights {
		activePods[pod] = struct{}{}
		podScores[pod] = w
	}

	// Iterate left-to-right (prefix matching)
	for i := 1; i < len(keys); i++ {
		if len(activePods) == 0 {
			break
		}

		clear(curWeights)
		entries := filterByGroup(keyToPods[keys[i]], groupID)
		// TODO(tmp): Debug logging - REMOVE after debugging
		if i < 3 { // Only log first few blocks to avoid spam
			fmt.Printf("[TMP-DEBUG] scoreFullAttentionGroup: blockIdx=%d, groupID=%d, totalEntries=%d, filteredEntries=%d, activePods=%d\n",
				i, groupID, len(keyToPods[keys[i]]), len(entries), len(activePods))
		}
		fillMaxWeights(curWeights, entries, s.MediumWeights)

		for pod := range activePods {
			if w, exists := curWeights[pod]; exists {
				podScores[pod] += w
			} else {
				delete(activePods, pod)
			}
		}
	}

	return podScores
}

// scoreSlidingWindowGroup scores sliding window attention group with suffix matching (right-to-left).
func (s *HybridPrefixCacheScorer) scoreSlidingWindowGroup(
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
	groupID int,
	windowSize int,
) map[string]float64 {
	podScores := make(map[string]float64)
	curWeights := make(map[string]float64)

	// Calculate the window range: only consider last windowSize blocks
	totalBlocks := len(keys)
	startIdx := 0
	if windowSize > 0 && totalBlocks > windowSize {
		startIdx = totalBlocks - windowSize
	}

	// TODO(tmp): Debug logging - REMOVE after debugging
	fmt.Printf("[TMP-DEBUG] scoreSlidingWindowGroup: groupID=%d, totalBlocks=%d, windowSize=%d, startIdx=%d\n",
		groupID, totalBlocks, windowSize, startIdx)

	// Start from the last block (rightmost in window)
	lastIdx := totalBlocks - 1
	lastEntries := filterByGroup(keyToPods[keys[lastIdx]], groupID)
	// TODO(tmp): Debug logging - REMOVE after debugging
	fmt.Printf("[TMP-DEBUG] scoreSlidingWindowGroup: blockIdx=%d (last), groupID=%d, totalEntries=%d, filteredEntries=%d\n",
		lastIdx, groupID, len(keyToPods[keys[lastIdx]]), len(lastEntries))
	fillMaxWeights(curWeights, lastEntries, s.MediumWeights)

	activePods := make(map[string]struct{}, len(curWeights))
	for pod, w := range curWeights {
		activePods[pod] = struct{}{}
		podScores[pod] = w
	}

	// Iterate right-to-left (suffix matching) within window
	for i := lastIdx - 1; i >= startIdx; i-- {
		if len(activePods) == 0 {
			break
		}

		clear(curWeights)
		entries := filterByGroup(keyToPods[keys[i]], groupID)
		// TODO(tmp): Debug logging - REMOVE after debugging
		if i > lastIdx-3 { // Only log last few blocks to avoid spam
			fmt.Printf("[TMP-DEBUG] scoreSlidingWindowGroup: blockIdx=%d, groupID=%d, totalEntries=%d, filteredEntries=%d, activePods=%d\n",
				i, groupID, len(keyToPods[keys[i]]), len(entries), len(activePods))
		}
		fillMaxWeights(curWeights, entries, s.MediumWeights)

		for pod := range activePods {
			if w, exists := curWeights[pod]; exists {
				podScores[pod] += w
			} else {
				delete(activePods, pod)
			}
		}
	}

	return podScores
}

// filterByGroup filters pod entries to only those containing the specified group ID.
func filterByGroup(entries []kvblock.PodEntry, groupID int) []kvblock.PodEntry {
	var filtered []kvblock.PodEntry
	for _, entry := range entries {
		if containsGroup(entry.StoredGroups, groupID) {
			filtered = append(filtered, entry)
		}
	}
	return filtered
}

// containsGroup checks if a group ID exists in the StoredGroups slice.
func containsGroup(storedGroups []int, groupID int) bool {
	for _, g := range storedGroups {
		if g == groupID {
			return true
		}
	}
	return false
}
