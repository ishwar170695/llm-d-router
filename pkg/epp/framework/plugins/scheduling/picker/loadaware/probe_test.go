/*
Copyright 2026 The Kubernetes Authors.

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

package loadaware

// Verification probes for the numbers cited in the #1598 discussion.
// Not part of the PR; kept separate so the findings are reproducible.

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/jellydator/ttlcache/v3"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// feedCompletion seeds the request cache and drives one ResponseBody completion.
func feedCompletion(p *LoadAwarePicker, id string, inputTokens, completionTokens int, duration time.Duration) {
	p.requestCache.Set(id, &requestRecord{dispatchTime: time.Now().Add(-duration)}, ttlcache.DefaultTTL)
	perPrompt := []uint32{}
	for i := range inputTokens {
		perPrompt = append(perPrompt, uint32(i))
	}
	req := &fwksched.InferenceRequest{
		RequestID: id,
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{perPrompt}},
		},
	}
	resp := &fwkrc.Response{
		EndOfStream: true,
		Usage:       fwkrh.Usage{CompletionTokens: completionTokens},
	}
	epMeta := &fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "ep1", Namespace: "default"}}
	p.ResponseBody(context.Background(), req, resp, epMeta)
}

// TestProbe_CapacityEMA feeds varied completions (short/long, fast/slow) and prints
// the resulting EMAs, maxConcurrency, and capacityFactor at 0/1/5 in-flight requests.
func TestProbe_CapacityEMA(t *testing.T) {
	p := newTestPicker(t)

	// Varied mix: {inputTokens, completionTokens, durationSeconds}
	mix := [][3]float64{
		{500, 20, 0.2},   // short, fast
		{500, 400, 2},    // medium
		{2000, 2000, 10}, // long, slow
		{200, 10, 0.1},   // short, fast
		{1000, 100, 1},
		{500, 900, 5},    // long output
		{3000, 100, 8},   // long input, short output
		{100, 50, 0.5},
		{800, 800, 6},
		{400, 40, 0.4},
		{1500, 300, 3},
		{250, 25, 0.3},
		{600, 600, 4},
		{900, 90, 0.9},
		{100, 1500, 12}, // very long output
		{2000, 20, 1},
		{700, 70, 0.7},
		{350, 35, 0.35},
		{1200, 120, 1.2},
		{50, 5, 0.05},
	}
	for i, m := range mix {
		feedCompletion(p, fmt.Sprintf("req-%d", i), int(m[0]), int(m[1]), time.Duration(m[2]*float64(time.Second)))
	}

	p.mu.RLock()
	st := p.state["default/ep1"]
	p.mu.RUnlock()
	if st == nil {
		t.Fatal("no state after completions")
	}

	maxConcurrency := st.capacityEMA * st.avgLatency / st.avgTokensPerReq
	fmt.Printf("\n=== PROBE A: capacity EMA after %d varied completions ===\n", len(mix))
	fmt.Printf("capacityEMA      = %12.4f (tokens/sec)\n", st.capacityEMA)
	fmt.Printf("avgLatency       = %12.4f (sec)\n", st.avgLatency)
	fmt.Printf("avgTokensPerReq  = %12.4f (tokens)\n", st.avgTokensPerReq)
	fmt.Printf("maxConcurrency   = %12.6f = capacityEMA*avgLatency/avgTokensPerReq\n", maxConcurrency)
	for _, inflight := range []float64{0, 1, 5} {
		f := 1.0 - inflight/maxConcurrency
		if f < 0 {
			f = 0
		}
		fmt.Printf("capacityFactor(@inflight=%v) = %.6f\n", inflight, f)
	}
	fmt.Printf("=== END PROBE A ===\n\n")
}

// TestProbe_PickerConcentration runs repeated picks over a pool of 4 identical
// endpoints, each with one in-flight request and identical EMA state, resetting
// pickBuckets/totalBuckets between picks so only the capacity factor acts.
func TestProbe_PickerConcentration(t *testing.T) {
	p := newTestPicker(t)

	const poolSize = 4
	names := []string{"ep1", "ep2", "ep3", "ep4"}
	var scored []*fwksched.ScoredEndpoint
	for _, n := range names {
		ep := fwksched.NewEndpoint(
			&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: n, Namespace: "default"}},
			&fwkdl.Metrics{RunningRequestsSize: 1}, // one in-flight request per endpoint
			nil,
		)
		scored = append(scored, &fwksched.ScoredEndpoint{Endpoint: ep, Score: 1.0})
		// EMA state identical across endpoints: maxConcurrency = 150*2/300 = 1.0
		st := &endpointLiveState{
			capacityEMA:     150.0,
			avgLatency:      2.0,
			avgTokensPerReq: 300.0,
		}
		// Pick keys state by NamespacedName.String(), e.g. "default/ep1".
		p.state["default/"+n] = st
	}

	const rounds = 20
	winners := make([]string, 0, rounds)
	for r := range rounds {
		// Reset pick distribution state between picks (isolate capacity effect).
		for _, st := range p.state {
			st.pickBuckets = bucketedCounter{}
			st.pendingRequests.Store(0)
		}
		p.totalBuckets = bucketedCounter{}

		result := p.Pick(context.Background(), scored)
		winner := result.TargetEndpoints[0].GetMetadata().NamespacedName.Name
		winners = append(winners, winner)

		if r == 0 {
			fmt.Printf("\n=== PROBE B: per-factor contributions before pick #1 (pool=%d, all 1 in-flight) ===\n", poolSize)
			for i, ep := range scored {
				key := ep.GetMetadata().NamespacedName.String()
				st := p.state[key]
				conc := p.concentrationFactor(st, 0, poolSize)
				cap_ := p.capacityFactor(ep, st)
				final := ep.Score * conc * cap_
				fmt.Printf("  %-4s rawScore=%.4f  concentrationFactor=%.6f  capacityFactor=%.6f  finalScore=%.10f\n",
					names[i], ep.Score, conc, cap_, final)
			}
			fmt.Printf("  winner of pick #1: %s\n", winner)
			fmt.Printf("  (pool=%d <= candidateCap=%d so sampleCandidates preserves original order)\n", poolSize, candidateCap)
		}
	}

	fmt.Printf("\nwinners across %d picks: %v\n", rounds, winners)
	uniform := true
	for _, w := range winners {
		if w != winners[0] {
			uniform = false
		}
	}
	fmt.Printf("first candidate (%s) wins every pick: %v\n", winners[0], uniform)
	fmt.Printf("=== END PROBE B ===\n\n")
}
