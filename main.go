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
	"github.com/pointlander/gradient/tf32"
)

func main() {
	rnd := rand.New(rand.NewSource(1))

	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}
	fisher := datum.Fisher

	others := tf32.NewSet()
	others.Add("data", 4, len(fisher))

	w := others.Weights[0]
	for _, data := range fisher {
		for _, measure := range data.Measures {
			w.X = append(w.X, float32(measure))
		}
	}

	context, quant :=
		make([]tf32.Context, tf32.FractionBits),
		make([]func(a tf32.Meta) tf32.Meta, tf32.FractionBits)
	for i := range context {
		context[i].Quantize = uint(i + 1)
		quant[i] = tf32.U(context[i].Quant)
	}

	width := 8
	set := tf32.NewSet()
	set.Add("aw", 4, width)
	set.Add("bw", width, width)
	set.Add("cw", width, 4)
	set.Add("ab", width, 1)
	set.Add("bb", width, 1)
	set.Add("cb", 4, 1)

	for _, w := range set.Weights[:3] {
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, float32(rnd.NormFloat64()*factor))
		}
	}

	for i := 3; i < len(set.Weights); i++ {
		set.Weights[i].X = set.Weights[i].X[:cap(set.Weights[i].X)]
	}

	deltas := make([][]float32, 0, 8)
	for _, p := range set.Weights {
		deltas = append(deltas, make([]float32, len(p.X)))
	}

	l1 := tf32.Softmax(tf32.Add(tf32.Mul(set.Get("aw"), others.Get("data")), set.Get("ab")))
	l2 := tf32.Softmax(quant[21](tf32.Add(tf32.Mul(set.Get("bw"), l1), set.Get("bb"))))
	distribution := quant[21](tf32.Add(tf32.Mul(set.Get("bw"), l1), set.Get("bb")))
	l3 := tf32.Add(tf32.Mul(set.Get("cw"), l2), set.Get("cb"))
	cost := tf32.Avg(tf32.Quadratic(l3, others.Get("data")))

	alpha, eta, iterations := float32(.1), float32(.1), 2048
	points := make(plotter.XYs, 0, iterations)
	i := 0
	for i < iterations {
		total := float32(0.0)
		set.Zero()
		others.Zero()

		total += tf32.Gradient(cost).X[0]
		sum := float32(0.0)
		for _, p := range set.Weights {
			for _, d := range p.D {
				sum += d * d
			}
		}
		norm := float32(math.Sqrt(float64(sum)))
		scaling := float32(1.0)
		if norm > 1 {
			scaling = 1 / norm
		}

		for j, w := range set.Weights {
			for k, d := range w.D {
				deltas[j][k] = alpha*deltas[j][k] - eta*d*scaling
				set.Weights[j].X[k] += deltas[j][k]
			}
		}

		points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
		fmt.Println(i, total)
		if total < 1e-3 {
			break
		}
		i++
	}

	distribution(func(a *tf32.V) bool {
		v := make(plotter.Values, 0, 8)
		for _, value := range a.X {
			v = append(v, float64(value))
		}

		p := plot.New()
		p.Title.Text = "Distribution"

		h, err := plotter.NewHist(v, 16)
		if err != nil {
			panic(err)
		}

		p.Add(h)

		err = p.Save(8*vg.Inch, 8*vg.Inch, "histogram.png")
		if err != nil {
			panic(err)
		}
		return true
	})

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

	v := make(plotter.Values, 0, 8)
	for _, value := range w.X {
		v = append(v, float64(value))
	}

	p = plot.New()
	p.Title.Text = "Distribution"

	h, err := plotter.NewHist(v, 16)
	if err != nil {
		panic(err)
	}

	p.Add(h)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "input_histogram.png")
	if err != nil {
		panic(err)
	}
}
