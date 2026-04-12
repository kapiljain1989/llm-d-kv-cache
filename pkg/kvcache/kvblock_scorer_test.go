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

package kvcache_test

import (
	"context"
	"testing"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/stretchr/testify/assert"
)

const (
	testModelName = "test-model"
	podA          = "pod-a"
	podB          = "pod-b"
)

// TestLongestPrefixScorer verifies scoring based on consecutive block hits from the start.
func TestLongestPrefixScorer(t *testing.T) {
	mediumWeights := map[string]float64{
		"gpu": 1.0,
		"cpu": 0.5,
	}

	scorer := &kvcache.LongestPrefixScorer{
		MediumWeights: mediumWeights,
	}
	blockKeys := int64KeysToKVBlockKeys([]uint64{1001, 1002, 1003, 1004, 1005, 1006})

	hitmap := map[kvblock.BlockHash][]kvblock.PodEntry{
		1001: {{PodIdentifier: podA, DeviceTier: "gpu"}},
		1002: {{PodIdentifier: podA, DeviceTier: "gpu"}},
		1003: {
			{PodIdentifier: podA, DeviceTier: "gpu"},
			{PodIdentifier: podA, DeviceTier: "cpu"},
		},
		1004: {{PodIdentifier: podB, DeviceTier: "cpu"}},
		1005: {{PodIdentifier: podB, DeviceTier: "cpu"}},
		1006: {{PodIdentifier: podA, DeviceTier: "gpu"}},
	}

	expected := map[string]float64{
		podA: 3.0,
		podB: 0.0,
	}

	scored, err := scorer.Score(context.Background(), blockKeys, hitmap, testModelName)
	assert.NoError(t, err)
	for pod, score := range scored {
		assert.InDelta(t, expected[pod], score, 0.0001)
	}
}

func TestLongestPrefixScorerDifferentTiers(t *testing.T) {
	mediumWeights := map[string]float64{
		"gpu": 1.0,
		"cpu": 0.5,
	}

	scorer := &kvcache.LongestPrefixScorer{
		MediumWeights: mediumWeights,
	}
	blockKeys := int64KeysToKVBlockKeys([]uint64{1001, 1002, 1003, 1004, 1005, 1006})

	hitmap := map[kvblock.BlockHash][]kvblock.PodEntry{
		1001: {{PodIdentifier: podA, DeviceTier: "gpu"}},
		1002: {{PodIdentifier: podA, DeviceTier: "gpu"}},
		1003: {{PodIdentifier: podA, DeviceTier: "cpu"}},
		1004: {{PodIdentifier: podB, DeviceTier: "cpu"}},
		1005: {{PodIdentifier: podB, DeviceTier: "cpu"}},
		1006: {{PodIdentifier: podA, DeviceTier: "gpu"}},
	}

	expected := map[string]float64{
		podA: 2.5,
		podB: 0.0,
	}

	scored, err := scorer.Score(context.Background(), blockKeys, hitmap, testModelName)
	assert.NoError(t, err)
	for pod, score := range scored {
		assert.InDelta(t, expected[pod], score, 0.0001)
	}
}

func int64KeysToKVBlockKeys(keys []uint64) []kvblock.BlockHash {
	kvKeys := make([]kvblock.BlockHash, len(keys))
	for i, key := range keys {
		kvKeys[i] = kvblock.BlockHash(key)
	}
	return kvKeys
}

// TestHybridPrefixCacheScorer tests the HybridPrefixMatch scorer for HMA models.
func TestHybridPrefixCacheScorer(t *testing.T) {
	tests := []struct {
		name           string
		modelConfig    *kvcache.ModelConfig
		keys           []kvblock.BlockHash
		keyToPods      map[kvblock.BlockHash][]kvblock.PodEntry
		expectedScores map[string]float64
	}{
		{
			name: "FullAttentionOnly_PrefixMatching",
			modelConfig: &kvcache.ModelConfig{
				Name:  "TestModel",
				IsHMA: true,
				AttentionGroups: []kvcache.AttentionGroupConfig{
					{
						GroupID:       0,
						AttentionType: kvcache.AttentionTypeFull,
						BlockSize:     64,
					},
				},
			},
			keys: []kvblock.BlockHash{100, 101, 102},
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				100: {
					{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: []int{0}},
					{PodIdentifier: "podB", DeviceTier: "gpu", StoredGroups: []int{0}},
				},
				101: {
					{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: []int{0}},
				},
				102: {
					{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: []int{0}},
				},
			},
			expectedScores: map[string]float64{
				"podA": 3.0 * 100000.0, // has all 3 blocks (fullAttentionMultiplier)
				"podB": 1.0 * 100000.0, // has only first block
			},
		},
		{
			name: "SlidingWindowOnly_SuffixMatching",
			modelConfig: &kvcache.ModelConfig{
				Name:  "TestModel",
				IsHMA: true,
				AttentionGroups: []kvcache.AttentionGroupConfig{
					{
						GroupID:           1,
						AttentionType:     kvcache.AttentionTypeSlidingWindow,
						BlockSize:         64,
						SlidingWindowSize: 2,
					},
				},
			},
			keys: []kvblock.BlockHash{100, 101, 102, 103, 104},
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				100: {
					{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: []int{1}},
				},
				101: {
					{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: []int{1}},
				},
				102: {
					{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: []int{1}},
				},
				103: {
					{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: []int{1}},
					{PodIdentifier: "podB", DeviceTier: "gpu", StoredGroups: []int{1}},
				},
				104: {
					{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: []int{1}},
					{PodIdentifier: "podB", DeviceTier: "gpu", StoredGroups: []int{1}},
				},
			},
			expectedScores: map[string]float64{
				"podA": 2.0, // has last 2 blocks (103, 104) within window
				"podB": 2.0, // has last 2 blocks (103, 104) within window
			},
		},
		{
			name: "HybridModel_FullAndSlidingWindow",
			modelConfig: &kvcache.ModelConfig{
				Name:  "DeepSeek-V3",
				IsHMA: true,
				AttentionGroups: []kvcache.AttentionGroupConfig{
					{
						GroupID:       0,
						AttentionType: kvcache.AttentionTypeFull,
						BlockSize:     64,
					},
					{
						GroupID:           1,
						AttentionType:     kvcache.AttentionTypeSlidingWindow,
						BlockSize:         64,
						SlidingWindowSize: 2,
					},
				},
			},
			keys: []kvblock.BlockHash{100, 101, 102},
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				100: {
					{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: []int{0, 1}},
					{PodIdentifier: "podB", DeviceTier: "gpu", StoredGroups: []int{0, 1}},
				},
				101: {
					{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: []int{0, 1}},
					{PodIdentifier: "podB", DeviceTier: "gpu", StoredGroups: []int{0, 1}},
				},
				102: {
					{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: []int{0, 1}},
				},
			},
			expectedScores: map[string]float64{
				// podA: full=3 blocks * 100000 + swa=2 blocks (last 2: 101, 102)
				"podA": 3.0*100000.0 + 2.0,
				// podB: full=2 blocks * 100000 + swa=0 (doesn't have block 102, chain breaks)
				"podB": 2.0 * 100000.0,
			},
		},
		{
			name: "FullAttentionDominates_EvenWithLowerSWA",
			modelConfig: &kvcache.ModelConfig{
				Name:  "TestModel",
				IsHMA: true,
				AttentionGroups: []kvcache.AttentionGroupConfig{
					{
						GroupID:       0,
						AttentionType: kvcache.AttentionTypeFull,
						BlockSize:     64,
					},
					{
						GroupID:           1,
						AttentionType:     kvcache.AttentionTypeSlidingWindow,
						BlockSize:         64,
						SlidingWindowSize: 10,
					},
				},
			},
			keys: []kvblock.BlockHash{100, 101},
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				100: {
					{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: []int{0}},
					{PodIdentifier: "podB", DeviceTier: "gpu", StoredGroups: []int{1}},
				},
				101: {
					{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: []int{0}},
					{PodIdentifier: "podB", DeviceTier: "gpu", StoredGroups: []int{1}},
				},
			},
			expectedScores: map[string]float64{
				// podA: full=2 blocks * 100000, podB: swa=2 blocks
				// podA should win due to full attention multiplier
				"podA": 2.0 * 100000.0,
				"podB": 2.0,
			},
		},
		{
			name: "GroupFiltering_OnlyScoreRelevantGroups",
			modelConfig: &kvcache.ModelConfig{
				Name:  "TestModel",
				IsHMA: true,
				AttentionGroups: []kvcache.AttentionGroupConfig{
					{
						GroupID:       0,
						AttentionType: kvcache.AttentionTypeFull,
						BlockSize:     64,
					},
				},
			},
			keys: []kvblock.BlockHash{100, 101},
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				100: {
					{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: []int{0}},
					{PodIdentifier: "podB", DeviceTier: "gpu", StoredGroups: []int{1}}, // group 1, should be filtered out
				},
				101: {
					{PodIdentifier: "podA", DeviceTier: "gpu", StoredGroups: []int{0}},
				},
			},
			expectedScores: map[string]float64{
				"podA": 2.0 * 100000.0, // only podA has group 0
				// podB not in results because it doesn't have group 0
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			registry := kvcache.NewModelRegistry([]*kvcache.ModelConfig{tt.modelConfig})

			config := &kvcache.KVBlockScorerConfig{
				ScoringStrategy: kvcache.HybridPrefixMatch,
				ModelRegistry:   registry,
				BackendConfigs: []*kvcache.KVCacheBackendConfig{
					{Name: "gpu", Weight: 1.0},
				},
			}

			scorer, err := kvcache.NewKVBlockScorer(config)
			assert.NoError(t, err)
			assert.Equal(t, kvcache.HybridPrefixMatch, scorer.Strategy())

			ctx := context.Background()
			scores, err := scorer.Score(ctx, tt.keys, tt.keyToPods, tt.modelConfig.Name)
			assert.NoError(t, err)

			assert.Equal(t, tt.expectedScores, scores,
				"Scores mismatch for test case: %s", tt.name)
		})
	}
}

// TestScorerSelection tests automatic scorer selection based on model configuration.
func TestScorerSelection(t *testing.T) {
	tests := []struct {
		name             string
		modelConfigs     []*kvcache.ModelConfig
		expectedStrategy kvcache.KVScoringStrategy
	}{
		{
			name:             "NoModels_UseSimpleScorer",
			modelConfigs:     nil,
			expectedStrategy: kvcache.LongestPrefixMatch,
		},
		{
			name: "OnlySimpleModels_UseSimpleScorer",
			modelConfigs: []*kvcache.ModelConfig{
				{Name: "Qwen/Qwen3-8B", IsHMA: false},
				{Name: "Llama-3-8B", IsHMA: false},
			},
			expectedStrategy: kvcache.LongestPrefixMatch,
		},
		{
			name: "OnlyHMAModels_UseHybridScorer",
			modelConfigs: []*kvcache.ModelConfig{
				{
					Name:  "DeepSeek-V3",
					IsHMA: true,
					AttentionGroups: []kvcache.AttentionGroupConfig{
						{GroupID: 0, AttentionType: kvcache.AttentionTypeFull, BlockSize: 64},
						{
							GroupID:           1,
							AttentionType:     kvcache.AttentionTypeSlidingWindow,
							BlockSize:         64,
							SlidingWindowSize: 4096,
						},
					},
				},
			},
			expectedStrategy: kvcache.HybridPrefixMatch,
		},
		{
			name: "MixedModels_UseHybridScorer",
			modelConfigs: []*kvcache.ModelConfig{
				{Name: "Qwen/Qwen3-8B", IsHMA: false},
				{
					Name:  "DeepSeek-V3",
					IsHMA: true,
					AttentionGroups: []kvcache.AttentionGroupConfig{
						{GroupID: 0, AttentionType: kvcache.AttentionTypeFull, BlockSize: 64},
					},
				},
			},
			expectedStrategy: kvcache.HybridPrefixMatch,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Simulate the logic from NewKVCacheIndexer
			var modelRegistry *kvcache.ModelRegistry
			if len(tt.modelConfigs) > 0 {
				modelRegistry = kvcache.NewModelRegistry(tt.modelConfigs)
			} else {
				modelRegistry = kvcache.NewDefaultModelRegistry()
			}

			// Create scorer config
			scorerConfig := kvcache.DefaultKVBlockScorerConfig()

			// Auto-select scoring strategy (same logic as in indexer.go)
			hasHMA := false
			for _, cfg := range tt.modelConfigs {
				if cfg.IsHMA {
					hasHMA = true
					break
				}
			}

			if hasHMA {
				scorerConfig.ScoringStrategy = kvcache.HybridPrefixMatch
				scorerConfig.ModelRegistry = modelRegistry
			} else {
				scorerConfig.ScoringStrategy = kvcache.LongestPrefixMatch
			}

			// Verify strategy selection
			assert.Equal(t, tt.expectedStrategy, scorerConfig.ScoringStrategy,
				"Strategy mismatch for test case: %s", tt.name)

			// Create scorer and verify it works
			scorer, err := kvcache.NewKVBlockScorer(scorerConfig)
			assert.NoError(t, err)
			assert.Equal(t, tt.expectedStrategy, scorer.Strategy())
		})
	}
}

// TestHybridScorerWithoutModelRegistry verifies error handling.
func TestHybridScorerWithoutModelRegistry(t *testing.T) {
	config := &kvcache.KVBlockScorerConfig{
		ScoringStrategy: kvcache.HybridPrefixMatch,
		ModelRegistry:   nil, // missing model registry
		BackendConfigs:  kvcache.DefaultKVCacheBackendConfig(),
	}

	_, err := kvcache.NewKVBlockScorer(config)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "model registry required")
}
