// Copyright 2022 The Diffusion Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"image/color"
	"io"
	"math"
	"math/rand"
	"os"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/datum/iris"
	"github.com/pointlander/gradient/tf32"
)

var (
	// FlagGaussian is gaussian mode
	FlagGaussian = flag.Bool("gaussian", false, "gaussian mode")
	// FlagQuantize is quantization mode
	FlagQuantize = flag.Bool("quantize", false, "quantization mode")
)

type Mode int

const (
	ModeNone Mode = iota
	ModeRaw
	ModeOrthogonality
	ModeParallel
	ModeMixed
	ModeEntropy
	ModeVariance
	NumberOfModes
)

func (m Mode) String() string {
	switch m {
	case ModeNone:
		return "none"
	case ModeRaw:
		return "raw"
	case ModeOrthogonality:
		return "orthogonality"
	case ModeMixed:
		return "mixed"
	case ModeParallel:
		return "parallel"
	case ModeEntropy:
		return "entropy"
	case ModeVariance:
		return "variance"
	}
	return "unknown"
}

func printTable(out io.Writer, headers []string, rows [][]string) {
	sizes := make([]int, len(headers))
	for i, header := range headers {
		sizes[i] = len(header)
	}
	for _, row := range rows {
		for j, item := range row {
			if length := len(item); length > sizes[j] {
				sizes[j] = length
			}
		}
	}

	last := len(headers) - 1
	fmt.Fprintf(out, "| ")
	for i, header := range headers {
		fmt.Fprintf(out, "%s", header)
		spaces := sizes[i] - len(header)
		for spaces > 0 {
			fmt.Fprintf(out, " ")
			spaces--
		}
		fmt.Fprintf(out, " |")
		if i < last {
			fmt.Fprintf(out, " ")
		}
	}
	fmt.Fprintf(out, "\n| ")
	for i, header := range headers {
		dashes := len(header)
		if sizes[i] > dashes {
			dashes = sizes[i]
		}
		for dashes > 0 {
			fmt.Fprintf(out, "-")
			dashes--
		}
		fmt.Fprintf(out, " |")
		if i < last {
			fmt.Fprintf(out, " ")
		}
	}
	fmt.Fprintf(out, "\n")
	for _, row := range rows {
		fmt.Fprintf(out, "| ")
		last := len(row) - 1
		for i, entry := range row {
			spaces := sizes[i] - len(entry)
			fmt.Fprintf(out, "%s", entry)
			for spaces > 0 {
				fmt.Fprintf(out, " ")
				spaces--
			}
			fmt.Fprintf(out, " |")
			if i < last {
				fmt.Fprintf(out, " ")
			}
		}
		fmt.Fprintf(out, "\n")
	}
}

// Statistics captures statistics
type Statistics struct {
	Sum        float64
	SumSquared float64
	Count      int
}

// Add adds a statistic
func (s *Statistics) Add(value float64) {
	s.Sum += value
	s.SumSquared += value * value
	s.Count++
}

// StandardDeviation calculates the standard deviation
func (s Statistics) StandardDeviation() float64 {
	sum, count := s.Sum, float64(s.Count)
	return math.Sqrt((s.SumSquared - sum*sum/count) / count)
}

// Average calculates the average
func (s Statistics) Average() float64 {
	return s.Sum / float64(s.Count)
}

// String returns the statistics as a string`
func (s Statistics) String() string {
	return fmt.Sprintf("%f +- %f", s.Average(), s.StandardDeviation())
}

func main() {
	flag.Parse()

	if *FlagGaussian {
		Gaussian()
	} else if *FlagQuantize {
		Quantize()
	} else {
		Start()
	}
}

func Start() {
	rnd := rand.New(rand.NewSource(1))
	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}
	var stats [4]Statistics
	for _, data := range datum.Fisher {
		for i, measure := range data.Measures {
			stats[i].Add(measure)
		}
	}
	reduction := Process("", rnd, stats, 1, datum.Fisher)
	out, err := os.Create("results/result.md")
	if err != nil {
		panic(err)
	}
	defer out.Close()
	reduction.PrintTable(out, ModeRaw, 0)
}

func Process(lr string, rnd *rand.Rand, stats [4]Statistics, depth int, data []iris.Iris) *Reduction {
	name := fmt.Sprintf("%s%dnode", lr, depth)
	embeddings := Segment(rnd, stats, name, 4, 4, data)
	reduction := embeddings.VarianceReduction(1, 0, 0)
	if depth <= 0 {
		return reduction
	}
	var left []iris.Iris
	for _, embedding := range reduction.Left.Embeddings.Embeddings {
		left = append(left, embedding.Iris)
	}
	reduction.Left = Process("l", rnd, stats, depth-1, left)
	var right []iris.Iris
	for _, embedding := range reduction.Right.Embeddings.Embeddings {
		right = append(right, embedding.Iris)
	}
	reduction.Right = Process("r", rnd, stats, depth-1, right)
	return reduction
}

func Segment(rnd *rand.Rand, stats [4]Statistics, name string, size, width int, iris []iris.Iris) *Embeddings {
	others := tf32.NewSet()
	others.Add("input", 4, len(iris))
	others.Add("output", 4, len(iris))

	for _, w := range others.Weights {
		for _, data := range iris {
			for _, measure := range data.Measures {
				w.X = append(w.X, float32(measure))
			}
		}
	}
	inputs := others.Weights[0]

	train := func(name string, size, width int, input tf32.Meta) (tf32.Meta, tf32.Meta) {
		fmt.Printf("\n")
		fmt.Println(name)
		set := tf32.NewSet()
		set.Add("aw", size, width)
		set.Add("bw", width, 4)
		set.Add("ab", width, 1)
		set.Add("bb", 4, 1)

		for _, w := range set.Weights[:2] {
			factor := math.Sqrt(2.0 / float64(w.S[0]))
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, float32(rnd.NormFloat64()*factor))
			}
		}

		for i := 2; i < len(set.Weights); i++ {
			set.Weights[i].X = set.Weights[i].X[:cap(set.Weights[i].X)]
		}

		deltas := make([][]float32, 0, 8)
		for _, p := range set.Weights {
			deltas = append(deltas, make([]float32, len(p.X)))
		}

		l1 := tf32.TanH(tf32.Add(tf32.Mul(set.Get("aw"), input), set.Get("ab")))
		l2 := tf32.Add(tf32.Mul(set.Get("bw"), l1), set.Get("bb"))
		cost := tf32.Avg(tf32.Quadratic(l2, others.Get("output")))

		d := make([]float64, len(stats))
		for i, stat := range stats {
			d[i] = stat.StandardDeviation()
		}

		alpha, eta, iterations := float32(.1), float32(.1), 2048
		points := make(plotter.XYs, 0, iterations)
		i := 0
		for i < iterations {
			total := float32(0.0)
			set.Zero()
			others.Zero()

			if i == 128 || i == 2*128 || i == 3*128 || i == 4*128 {
				for j := range d {
					d[j] /= 10
				}
			}

			index := 0
			for _, data := range iris {
				for i, measure := range data.Measures {
					if d[i] == 0 {
						inputs.X[index] = float32(measure)
					} else {
						inputs.X[index] = float32(measure + rnd.NormFloat64()*d[i])
					}
					index++
				}
			}

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
			//fmt.Println(i, total)
			/*if total < .1 {
				break
			}*/
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

		err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("results/%s_gaussian_cost.png", name))
		if err != nil {
			panic(err)
		}

		return l1, cost
	}

	l1, cost := train(fmt.Sprintf("layer1_%s", name), 4, 16, others.Get("input"))
	l1, cost = train(fmt.Sprintf("layer2_%s", name), 16, 4, l1)

	index := 0
	for _, data := range iris {
		for _, measure := range data.Measures {
			inputs.X[index] = float32(measure)
			index++
		}
	}
	fmt.Println("cost", tf32.Gradient(cost).X[0])

	embeddings := Embeddings{
		Columns:    width,
		Network:    l1,
		CostCurve:  fmt.Sprintf("results/%s_gaussian_cost.png", name),
		Embeddings: make([]Embedding, 0, 8),
	}
	l1(func(a *tf32.V) bool {
		v := make(plotter.Values, 0, 8)
		for _, value := range a.X {
			v = append(v, float64(value))
		}
		data := make(map[string][]float64)
		stats := make([]Statistics, width)

		for i, entry := range iris {
			embedding := Embedding{
				Iris:     entry,
				Features: make([]float64, 0, width),
			}
			for j := 0; j < width; j++ {
				fmt.Printf("%f ", a.X[i*width+j])
				stats[j].Add(float64(a.X[i*width+j]))
				embedding.Features = append(embedding.Features, float64(a.X[i*width+j]))
			}
			embeddings.Embeddings = append(embeddings.Embeddings, embedding)
			fmt.Printf("%s\n", entry.Label)
		}

		indexes := make([]int, 0, 8)
		for i := range stats {
			if stats[i].StandardDeviation() > .1 {
				indexes = append(indexes, i)
			}
		}
		for i, entry := range iris {
			for _, j := range indexes {
				s := data[entry.Label]
				if s == nil {
					s = make([]float64, 0, 8)
				}
				s = append(s, float64(a.X[i*width+j]))
				data[entry.Label] = s
			}
		}

		p := plot.New()
		p.Title.Text = "Distribution"

		h, err := plotter.NewHist(v, 64)
		if err != nil {
			panic(err)
		}

		p.Add(h)

		err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("results/%s_gaussian_histogram.png", name))
		if err != nil {
			panic(err)
		}

		/*p = plot.New()

		p.Title.Text = "x vs y"
		p.X.Label.Text = "x"
		p.Y.Label.Text = "y"
		for label, entry := range data {
			r, c := len(entry)/len(indexes), len(indexes)
			ranks := mat.NewDense(r, c, entry)
			var pc stat.PC
			ok := pc.PrincipalComponents(ranks, nil)
			if !ok {
				panic("PrincipalComponents failed")
			}
			k := 1
			if c >= 2 {
				k = 2
			}
			var proj mat.Dense
			var vec mat.Dense
			pc.VectorsTo(&vec)
			slice := vec.Slice(0, c, 0, k)
			proj.Mul(ranks, slice)

			points := make(plotter.XYs, 0, 8)
			for i := 0; i < r; i++ {
				y := 0.0
				if c >= 2 {
					y = proj.At(i, 1)
				}
				points = append(points, plotter.XY{X: proj.At(i, 0), Y: y})
			}

			scatter, err := plotter.NewScatter(points)
			if err != nil {
				panic(err)
			}
			scatter.GlyphStyle.Radius = vg.Length(3)
			scatter.GlyphStyle.Shape = draw.CircleGlyph{}
			if label == "Iris-virginica" {
				scatter.Color = color.RGBA{0xFF, 0, 0, 0xFF}
			} else if label == "Iris-versicolor" {
				scatter.Color = color.RGBA{0, 0xFF, 0, 0xFF}
			} else if label == "Iris-setosa" {
				scatter.Color = color.RGBA{0, 0, 0xFF, 0xFF}
			}
			p.Add(scatter)
		}
		err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("results/%s_projection.png", name))
		if err != nil {
			panic(err)
		}*/
		return true
	})

	for _, stat := range stats {
		fmt.Println(stat.StandardDeviation())
	}

	return &embeddings
}

// Guassian is a guassian diffusion neural network
func Gaussian() {
	rnd := rand.New(rand.NewSource(1))

	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}
	fisher := datum.Fisher

	others := tf32.NewSet()
	others.Add("input", 4, len(fisher))
	others.Add("output", 4, len(fisher))

	var stats [4]Statistics

	for _, w := range others.Weights {
		for _, data := range fisher {
			for i, measure := range data.Measures {
				stats[i].Add(measure)
				w.X = append(w.X, float32(measure))
			}
		}
	}
	inputs := others.Weights[0]

	train := func(name string, size, width int, input tf32.Meta) tf32.Meta {
		fmt.Printf("\n")
		fmt.Println(name)
		set := tf32.NewSet()
		set.Add("aw", size, width)
		set.Add("bw", width, 4)
		set.Add("ab", width, 1)
		set.Add("bb", 4, 1)

		for _, w := range set.Weights[:2] {
			factor := math.Sqrt(2.0 / float64(w.S[0]))
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, float32(rnd.NormFloat64()*factor))
			}
		}

		for i := 2; i < len(set.Weights); i++ {
			set.Weights[i].X = set.Weights[i].X[:cap(set.Weights[i].X)]
		}

		deltas := make([][]float32, 0, 8)
		for _, p := range set.Weights {
			deltas = append(deltas, make([]float32, len(p.X)))
		}

		l1 := tf32.TanH(tf32.Add(tf32.Mul(set.Get("aw"), input), set.Get("ab")))
		l2 := tf32.Add(tf32.Mul(set.Get("bw"), l1), set.Get("bb"))
		cost := tf32.Avg(tf32.Quadratic(l2, others.Get("output")))

		d := make([]float64, len(stats))
		for i, stat := range stats {
			d[i] = stat.StandardDeviation()
		}

		alpha, eta, iterations := float32(.1), float32(.1), 2048
		points := make(plotter.XYs, 0, iterations)
		i := 0
		for i < iterations {
			total := float32(0.0)
			set.Zero()
			others.Zero()

			if i == 128 || i == 2*128 || i == 3*128 || i == 4*128 {
				for j := range d {
					d[j] /= 10
				}
			}

			index := 0
			for _, data := range fisher {
				for i, measure := range data.Measures {
					if d[i] == 0 {
						inputs.X[index] = float32(measure)
					} else {
						inputs.X[index] = float32(measure + rnd.NormFloat64()*d[i])
					}
					index++
				}
			}

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
			/*if total < .1 {
				break
			}*/
			i++
		}

		index := 0
		for _, data := range fisher {
			for _, measure := range data.Measures {
				inputs.X[index] = float32(measure)
				index++
			}
		}
		fmt.Println(tf32.Gradient(cost).X[0])

		l1(func(a *tf32.V) bool {
			v := make(plotter.Values, 0, 8)
			for _, value := range a.X {
				v = append(v, float64(value))
			}
			data := make(map[string][]float64)
			stats := make([]Statistics, width)
			embeddings := Embeddings{
				Columns:    width,
				Network:    l1,
				CostCurve:  fmt.Sprintf("results/%s_gaussian_cost.png", name),
				Embeddings: make([]Embedding, 0, 8),
			}
			for i, entry := range fisher {
				embedding := Embedding{
					Iris:     entry,
					Features: make([]float64, 0, width),
				}
				for j := 0; j < width; j++ {
					fmt.Printf("%f ", a.X[i*width+j])
					stats[j].Add(float64(a.X[i*width+j]))
					embedding.Features = append(embedding.Features, float64(a.X[i*width+j]))
				}
				embeddings.Embeddings = append(embeddings.Embeddings, embedding)
				fmt.Printf("%s\n", entry.Label)
			}
			reduction := embeddings.VarianceReduction(1, 0, 0)
			out, err := os.Create(fmt.Sprintf("results/result_%s.md", name))
			if err != nil {
				panic(err)
			}
			defer out.Close()
			reduction.PrintTable(out, ModeRaw, 0)

			indexes := make([]int, 0, 8)
			for i := range stats {
				if stats[i].StandardDeviation() > .1 {
					indexes = append(indexes, i)
				}
			}
			for i, entry := range fisher {
				for _, j := range indexes {
					s := data[entry.Label]
					if s == nil {
						s = make([]float64, 0, 8)
					}
					s = append(s, float64(a.X[i*width+j]))
					data[entry.Label] = s
				}
			}

			p := plot.New()
			p.Title.Text = "Distribution"

			h, err := plotter.NewHist(v, 64)
			if err != nil {
				panic(err)
			}

			p.Add(h)

			err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("results/%s_gaussian_histogram.png", name))
			if err != nil {
				panic(err)
			}

			p = plot.New()

			p.Title.Text = "x vs y"
			p.X.Label.Text = "x"
			p.Y.Label.Text = "y"
			for label, entry := range data {
				r, c := len(entry)/len(indexes), len(indexes)
				ranks := mat.NewDense(r, c, entry)
				var pc stat.PC
				ok := pc.PrincipalComponents(ranks, nil)
				if !ok {
					panic("PrincipalComponents failed")
				}
				k := 1
				if c >= 2 {
					k = 2
				}
				var proj mat.Dense
				var vec mat.Dense
				pc.VectorsTo(&vec)
				slice := vec.Slice(0, c, 0, k)
				proj.Mul(ranks, slice)

				points := make(plotter.XYs, 0, 8)
				for i := 0; i < r; i++ {
					y := 0.0
					if c >= 2 {
						y = proj.At(i, 1)
					}
					points = append(points, plotter.XY{X: proj.At(i, 0), Y: y})
				}

				scatter, err := plotter.NewScatter(points)
				if err != nil {
					panic(err)
				}
				scatter.GlyphStyle.Radius = vg.Length(3)
				scatter.GlyphStyle.Shape = draw.CircleGlyph{}
				if label == "Iris-virginica" {
					scatter.Color = color.RGBA{0xFF, 0, 0, 0xFF}
				} else if label == "Iris-versicolor" {
					scatter.Color = color.RGBA{0, 0xFF, 0, 0xFF}
				} else if label == "Iris-setosa" {
					scatter.Color = color.RGBA{0, 0, 0xFF, 0xFF}
				}
				p.Add(scatter)
			}
			err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("results/%s_projection.png", name))
			if err != nil {
				panic(err)
			}
			return true
		})

		fmt.Println(set.Weights[0].X)

		for _, stat := range stats {
			fmt.Println(stat.StandardDeviation())
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

		err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("results/%s_gaussian_cost.png", name))
		if err != nil {
			panic(err)
		}
		return l1
	}

	l1 := train("layer1", 4, 16, others.Get("input"))
	_ = l1
	//train("layer2", 16, 4, l1)
}

// Quantize is a quantization diffusion network
func Quantize() {
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

	l1 := tf32.TanH(tf32.Add(tf32.Mul(set.Get("aw"), others.Get("data")), set.Get("ab")))
	l2 := tf32.TanH(quant[20](tf32.Add(tf32.Mul(set.Get("bw"), l1), set.Get("bb"))))
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
		if total < .5 {
			break
		}
		i++
	}

	l2(func(a *tf32.V) bool {
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
