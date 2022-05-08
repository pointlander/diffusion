// Copyright 2022 The Diffusion Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/datum/iris"
	"github.com/pointlander/gradient/tf64"
)

func main() {
	rnd := rand.New(rand.NewSource(1))

	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}
	fisher := datum.Fisher

	others := tf64.NewSet()
	others.Add("data", 4, len(fisher))

	w := others.Weights[0]
	for _, data := range fisher {
		w.X = append(w.X, data.Measures...)
	}

	set := tf64.NewSet()
	set.Add("aw", 4, 8)
	set.Add("bw", 8, 4)
	set.Add("ab", 8, 1)
	set.Add("bb", 4, 1)

	for _, w := range set.Weights[:2] {
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, rnd.NormFloat64()*factor)
		}
	}

	for i := 2; i < len(set.Weights); i++ {
		set.Weights[i].X = set.Weights[i].X[:cap(set.Weights[i].X)]
	}

	deltas := make([][]float64, 0, 8)
	for _, p := range set.Weights {
		deltas = append(deltas, make([]float64, len(p.X)))
	}

	l1 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("aw"), others.Get("data")), set.Get("ab")))
	l2 := tf64.Add(tf64.Mul(set.Get("bw"), l1), set.Get("bb"))
	cost := tf64.Avg(tf64.Quadratic(l2, others.Get("data")))

	alpha, eta, iterations := .1, .1, 1024
	points := make(plotter.XYs, 0, iterations)
	i := 0
	for i < iterations {
		total := 0.0
		set.Zero()
		others.Zero()

		total += tf64.Gradient(cost).X[0]
		sum := 0.0
		for _, p := range set.Weights {
			for _, d := range p.D {
				sum += d * d
			}
		}
		norm := math.Sqrt(sum)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}

		for j, w := range set.Weights {
			for k, d := range w.D {
				deltas[j][k] = alpha*deltas[j][k] - eta*d*scaling
				set.Weights[j].X[k] += deltas[j][k]
			}
		}

		points = append(points, plotter.XY{X: float64(i), Y: total})
		fmt.Println(i, total)
		if total < 1e-3 {
			break
		}
		i++
	}

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "cost.png")
	if err != nil {
		panic(err)
	}
}
