package spd

import (
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

type MatrixSPD struct {
	LearningRate float64
	Iterations   int
	RandomSeed   int64
}

func NewMatrixSPD(learningRate float64, iterations int) *MatrixSPD {
	return &MatrixSPD{
		LearningRate: learningRate,
		Iterations:   iterations,
		RandomSeed:   time.Now().UnixNano(),
	}
}

func (m *MatrixSPD) Decompose(data *mat.Dense, rank int) (*mat.Dense, *mat.Dense) {
	rand.Seed(m.RandomSeed)

	rows, cols := data.Dims()

	// Initialize random matrices
	factorU := mat.NewDense(rows, rank, nil)
	factorV := mat.NewDense(rank, cols, nil)

	randomizeMatrix(factorU)
	randomizeMatrix(factorV)

	// Stochastic gradient descent
	for iter := 0; iter < m.Iterations; iter++ {
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				val := data.At(i, j)
				if math.IsNaN(val) {
					continue // Skip missing values
				}

				// Compute prediction
				var prediction float64
				for k := 0; k < rank; k++ {
					prediction += factorU.At(i, k) * factorV.At(k, j)
				}

				err := val - prediction

				// Update factors
				for k := 0; k < rank; k++ {
					uik := factorU.At(i, k)
					vkj := factorV.At(k, j)

					factorU.Set(i, k, uik+m.LearningRate*(err*vkj))
					factorV.Set(k, j, vkj+m.LearningRate*(err*uik))
				}
			}
		}
	}

	return factorU, factorV
}

func randomizeMatrix(m *mat.Dense) {
	r, c := m.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			m.Set(i, j, rand.NormFloat64())
		}
	}
}
